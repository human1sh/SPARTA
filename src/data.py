"""Data ingestion, feature engineering, and PyTorch Dataset wrappers."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import torch
from torch.utils.data import DataLoader, Dataset


class ConjunctionDataset(Dataset):
    """PyTorch Dataset wrapper for satellite telemetry tensors."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def prepare_data_pipeline(
    csv_path: str, batch_size: int = 256, seed: int = 42
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    PowerTransformer,
    List[str],
    Tuple[np.ndarray, np.ndarray],
]:
    """Ingests raw CSV, encodes categorical variables, engineers astrodynamic features,
    applies power scaling, and returns DataLoaders along with test arrays.
    """
    df = pd.read_csv(csv_path)
    target_col = "avoidance_maneuver_executed"

    # --- Categorical One-Hot Encoding ---
    categorical_cols = ["primary_constellation", "secondary_object_type"]
    present_cat_cols = [c for c in categorical_cols if c in df.columns]
    if present_cat_cols:
        df = pd.get_dummies(df, columns=present_cat_cols, drop_first=True)

    # --- Basic Log Transforms ---
    if "miss_distance_total_m" in df.columns:
        df["log_miss_total"] = np.log1p(df["miss_distance_total_m"])
    if "collision_probability_raw" in df.columns:
        df["log_raw_prob"] = np.log1p(df["collision_probability_raw"] * 1e6)

    # --- Astrodynamic Interaction Engineering ---
    if all(
        c in df.columns
        for c in [
            "solar_flux_f10_7_index",
            "orbit_altitude_km",
            "collision_probability_raw",
        ]
    ):
        df["drag_risk_proxy"] = (
            df["solar_flux_f10_7_index"] / df["orbit_altitude_km"]
        ) * df["collision_probability_raw"]

    if all(
        c in df.columns
        for c in ["collision_probability_raw", "time_to_closest_approach_hours"]
    ):
        df["time_decay_risk"] = df["collision_probability_raw"] / (
            df["time_to_closest_approach_hours"] + 1.0
        )

    # --- Advanced Physical & Covariance Proxies ---
    if all(
        c in df.columns
        for c in [
            "radial_miss_distance_m",
            "in_track_miss_distance_m",
            "cross_track_miss_distance_m",
            "mahalanobis_covariance_miss_distance",
        ]
    ):
        df["covariance_ellipsoid_proxy"] = (
            df["radial_miss_distance_m"]
            * df["in_track_miss_distance_m"]
            * df["cross_track_miss_distance_m"]
        ) / (df["mahalanobis_covariance_miss_distance"] + 1e-6)

    if all(
        c in df.columns
        for c in [
            "atmospheric_density_kg_m3",
            "solar_flux_f10_7_index",
            "solar_storm_drag_amplification_factor",
        ]
    ):
        df["space_weather_drag_impact"] = (
            df["atmospheric_density_kg_m3"]
            * df["solar_flux_f10_7_index"]
            * df["solar_storm_drag_amplification_factor"]
        )

    # --- Build Full Feature List ---
    base_candidates = [
        "orbit_altitude_km",
        "orbit_inclination_deg",
        "time_to_closest_approach_hours",
        "miss_distance_total_m",
        "radial_miss_distance_m",
        "in_track_miss_distance_m",
        "cross_track_miss_distance_m",
        "relative_velocity_km_s",
        "collision_probability_raw",
        "hard_body_radius_sum_m",
        "solar_flux_f10_7_index",
        "geomagnetic_ap_index",
        "atmospheric_density_kg_m3",
        "orbital_decay_rate_m_day",
        "kessler_syndrome_fragmentation_index",
        "solar_storm_drag_amplification_factor",
        "mahalanobis_covariance_miss_distance",
        "confidence_score",
        "log_miss_total",
        "log_raw_prob",
        "drag_risk_proxy",
        "time_decay_risk",
        "covariance_ellipsoid_proxy",
        "space_weather_drag_impact",
    ]

    # Include newly generated dummy columns dynamically
    dummy_cols = [
        c
        for c in df.columns
        if c.startswith("primary_constellation_")
        or c.startswith("secondary_object_type_")
    ]
    candidate_features = base_candidates + dummy_cols

    feature_cols = [c for c in candidate_features if c in df.columns]
    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Stratified Splits (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    scaler = PowerTransformer(method="yeo-johnson")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_loader = DataLoader(
        ConjunctionDataset(X_train_scaled, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ConjunctionDataset(X_val_scaled, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ConjunctionDataset(X_test_scaled, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        feature_cols,
        (X_test_scaled, y_test),
    )
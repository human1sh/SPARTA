"""Inference script for predicting collision maneuver risk on telemetry CSVs."""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml

from src.models import AstrodynamicPINN, TemperatureScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers non-linear astrodynamic features to match model inputs."""
    df_eng = df.copy()

    if "miss_distance_total_m" in df_eng.columns:
        df_eng["log_miss_total"] = np.log1p(df_eng["miss_distance_total_m"])
    if "collision_probability_raw" in df_eng.columns:
        df_eng["log_raw_prob"] = np.log1p(df_eng["collision_probability_raw"] * 1e6)

    if all(
        c in df_eng.columns
        for c in [
            "solar_flux_f10_7_index",
            "orbit_altitude_km",
            "collision_probability_raw",
        ]
    ):
        df_eng["drag_risk_proxy"] = (
            df_eng["solar_flux_f10_7_index"] / df_eng["orbit_altitude_km"]
        ) * df_eng["collision_probability_raw"]

    if all(
        c in df_eng.columns
        for c in ["collision_probability_raw", "time_to_closest_approach_hours"]
    ):
        df_eng["time_decay_risk"] = df_eng["collision_probability_raw"] / (
            df_eng["time_to_closest_approach_hours"] + 1.0
        )

    if all(
        c in df_eng.columns
        for c in [
            "radial_miss_distance_m",
            "in_track_miss_distance_m",
            "cross_track_miss_distance_m",
            "mahalanobis_covariance_miss_distance",
        ]
    ):
        df_eng["covariance_ellipsoid_proxy"] = (
            df_eng["radial_miss_distance_m"]
            * df_eng["in_track_miss_distance_m"]
            * df_eng["cross_track_miss_distance_m"]
        ) / (df_eng["mahalanobis_covariance_miss_distance"] + 1e-6)

    if all(
        c in df_eng.columns
        for c in [
            "atmospheric_density_kg_m3",
            "solar_flux_f10_7_index",
            "solar_storm_drag_amplification_factor",
        ]
    ):
        df_eng["space_weather_drag_impact"] = (
            df_eng["atmospheric_density_kg_m3"]
            * df_eng["solar_flux_f10_7_index"]
            * df_eng["solar_storm_drag_amplification_factor"]
        )

    return df_eng


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Collision Avoidance Risk.")
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/predictions.csv",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/pinn_model.pt",
        help="Path to trained PyTorch model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_df = pd.read_csv(args.input_csv)
    df = engineer_features(raw_df)

    candidate_features = [
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
    feature_cols = [c for c in candidate_features if c in df.columns]

    model = AstrodynamicPINN(
        input_dim=len(feature_cols),
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    temp_scaler = TemperatureScaler().to(device)

    x_tensor = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
    with torch.no_grad():
        raw_logits = model(x_tensor)
        calibrated_logits = temp_scaler(raw_logits)
        predicted_probs = torch.sigmoid(calibrated_logits).cpu().numpy().flatten()

    raw_df["predicted_maneuver_probability"] = predicted_probs
    raw_df["predicted_avoidance_required"] = (predicted_probs >= 0.5).astype(int)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    raw_df.to_csv(args.output_csv, index=False)
    print(f"Inference complete. Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
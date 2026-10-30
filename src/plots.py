"""Module for generating diagnostic, physics constraint, and performance plots."""

import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils import calculate_ece


def plot_calibration_and_roc(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    save_path: str = "outputs/01_calibration_and_roc.png",
) -> None:
    """Generates dual-panel ROC curve and Reliability Diagram."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: ROC Curve
    auc_val = roc_auc_score(y_true, calibrated_probs)
    fpr, tpr, _ = roc_curve(y_true, calibrated_probs)
    axes[0].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2.5,
        label=f"Residual PINN (AUC = {auc_val:.4f})",
    )
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve: Refined Collision Risk Classification")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Reliability Diagram
    ax2 = axes[1]
    ax2_hist = ax2.twinx()

    ax2_hist.hist(
        calibrated_probs,
        bins=10,
        range=(0, 1),
        alpha=0.15,
        color="gray",
        edgecolor="black",
    )
    ax2_hist.set_ylabel("Sample Count per Bin (Log Scale)", color="gray")
    ax2_hist.set_yscale("log")

    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, raw_probs, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, calibrated_probs, n_bins=10)

    raw_ece = calculate_ece(raw_probs, y_true)
    calibrated_ece = calculate_ece(calibrated_probs, y_true)

    ax2.plot(
        mean_pred_raw,
        frac_pos_raw,
        "s-",
        color="red",
        label=f"Uncalibrated (ECE = {raw_ece:.3f})",
    )
    ax2.plot(
        mean_pred_cal,
        frac_pos_cal,
        "s-",
        color="green",
        label=f"Calibrated (ECE = {calibrated_ece:.3f})",
    )
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives")
    ax2.set_title("Reliability Diagram (With Bin Sample Densities)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_physics_monotonicity(
    model: nn.Module,
    sample_inputs: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    save_path: str = "outputs/02_physics_monotonicity.png",
) -> None:
    """Verifies d(P_risk)/d(miss_distance) <= 0 by sweeping miss distance."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    miss_idx = feature_names.index("miss_distance_total_m")

    model.eval()
    fig, ax = plt.subplots(figsize=(8, 5))

    sweep_range = np.linspace(-2.0, 2.0, 100)

    for i in range(min(5, len(sample_inputs))):
        base_sample = sample_inputs[i].copy()
        probs = []

        for val in sweep_range:
            base_sample[miss_idx] = val
            tensor_in = (
                torch.tensor(base_sample, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor_in)).item()
                probs.append(prob)

        ax.plot(sweep_range, probs, label=f"Conjunction Profile #{i+1}", lw=2)

    ax.set_xlabel("Normalized Miss Distance ($r_{\\mathrm{miss}}$)")
    ax.set_ylabel("Predicted Collision Risk ($P_{\\mathrm{risk}}$)")
    ax.set_title(
        "Physics Monotonicity Verification: $\\partial P_{\\mathrm{risk}} / \\partial r_{\\mathrm{miss}} \\leq 0$"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_trajectories(
    train_losses: List[float],
    physics_losses: List[float],
    save_path: str = "outputs/03_loss_trajectories.png",
) -> None:
    """Plots training BCE loss alongside the physics penalty decay curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Total Training Loss (BCE + Physics)", color=color)
    ax1.plot(epochs, train_losses, color=color, lw=2, label="Total Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel(
        "Physics Monotonicity Penalty ($\\mathcal{L}_{\\mathrm{physics}}$)",
        color=color,
    )
    ax2.plot(
        epochs,
        physics_losses,
        color=color,
        lw=2,
        linestyle="--",
        label="Physics Loss",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Training Convergence & Physics Penalty Decay Trajectories")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_space_weather_risk_proxy(
    solar_flux: np.ndarray,
    altitude: np.ndarray,
    predicted_probs: np.ndarray,
    save_path: str = "outputs/04_drag_risk_heatmap.png",
) -> None:
    """Scatter heatmap showing Solar Flux vs Altitude vs Predicted Collision Risk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        altitude,
        solar_flux,
        c=predicted_probs,
        cmap="coolwarm",
        alpha=0.7,
        edgecolors="none",
        s=30,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Predicted Collision Risk ($P_{\\mathrm{risk}}$)")

    ax.set_xlabel("Orbital Altitude (km)")
    ax.set_ylabel("Solar Flux ($F_{10.7}$ Index)")
    ax.set_title("Space Weather & Orbital Altitude Interaction Heatmap")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_probability_distribution(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    save_path: str = "outputs/05_risk_probability_distribution.png",
) -> None:
    """Overlaid KDE histogram comparing maneuver-required vs safe conjunction events."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        calibrated_probs[y_true == 0],
        bins=30,
        alpha=0.5,
        label="No Maneuver Executed",
        color="green",
        density=True,
    )
    ax.hist(
        calibrated_probs[y_true == 1],
        bins=30,
        alpha=0.5,
        label="Maneuver Required",
        color="red",
        density=True,
    )

    ax.axvline(
        0.5,
        color="black",
        linestyle="--",
        lw=1.5,
        label="Decision Threshold (0.5)",
    )
    ax.set_xlabel("Calibrated Collision Risk Probability ($P_{\\mathrm{risk}}$)")
    ax.set_ylabel("Density")
    ax.set_title("Test Sample Risk Probability Separability Distribution")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    model: nn.Module,
    sample_inputs: np.ndarray,
    feature_names: List[str],
    train_losses: List[float],
    physics_losses: List[float],
    solar_flux: np.ndarray,
    altitude: np.ndarray,
    device: torch.device,
) -> None:
    """Executes all figure generation functions sequentially."""
    print("\n--- Generating All Diagnostic & Physics Figures ---")
    plot_calibration_and_roc(y_true, raw_probs, calibrated_probs)
    plot_physics_monotonicity(model, sample_inputs, feature_names, device)
    plot_loss_trajectories(train_losses, physics_losses)
    plot_space_weather_risk_proxy(solar_flux, altitude, calibrated_probs)
    plot_probability_distribution(y_true, calibrated_probs)
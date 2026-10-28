"""Evaluation metrics, diagnostic plotting, and calibration utilities."""

import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def calculate_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) across probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(targets[in_bin])
            conf_in_bin = np.mean(probs[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
) -> Dict[str, float]:
    """Computes discrimination and reliability performance metrics."""
    preds_binary = (calibrated_probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, calibrated_probs)),
        "macro_f1": float(f1_score(y_true, preds_binary, average="macro")),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, preds_binary)
        ),
        "brier_score": float(brier_score_loss(y_true, calibrated_probs)),
        "raw_ece": calculate_ece(raw_probs, y_true),
        "calibrated_ece": calculate_ece(calibrated_probs, y_true),
    }
    return metrics


def plot_diagnostics(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    save_path: str = "outputs/figures/output.png",
) -> None:
    """Generates and saves dual-panel ROC and Calibration diagnostic plot."""
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

    # Panel 2: Reliability Diagram + Sample Density Histogram
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

    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_true, raw_probs, n_bins=10
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_true, calibrated_probs, n_bins=10
    )

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
    plt.savefig(save_path)
    plt.close()
    print(f"Diagnostic figure successfully saved to {save_path}")
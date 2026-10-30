"""Unified execution script for training, calibrating, evaluating, and plotting."""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data import prepare_data_pipeline
from src.loss import MonotonicityPhysicsLoss
from src.models import AstrodynamicPINN, TemperatureScaler
from src.plots import generate_all_plots
from src.utils import evaluate_predictions


def run_pipeline(config_path: str) -> None:
    """Executes the complete SciML pipeline end-to-end with progress prints."""
    print("🚀 [Step 0/7] Loading configuration file...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   --> Pipeline running on device: {device}")

    os.makedirs(config["outputs"]["output_dir"], exist_ok=True)
    os.makedirs(config["outputs"]["figures_dir"], exist_ok=True)
    print("   --> Output directories verified.")

    # 1. Prepare Data
    print("\n📊 [Step 1/7] Ingesting telemetry data and building PyTorch DataLoaders...")
    train_loader, val_loader, test_loader, _, feature_names, (x_test, y_test) = (
        prepare_data_pipeline(
            csv_path=config["data"]["csv_path"],
            batch_size=config["data"]["batch_size"],
            seed=config["data"]["seed"],
        )
    )
    print(f"   --> Complete! Features loaded: {len(feature_names)}")

    # 2. Initialize Model & Physics Loss
    print("\n⚙️ [Step 2/7] Initializing Astrodynamic PINN and Monotonicity Physics Loss...")
    model = AstrodynamicPINN(
        input_dim=len(feature_names),
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    physics_loss_fn = MonotonicityPhysicsLoss(
        miss_distance_idx=feature_names.index("miss_distance_total_m"),
        penalty_lambda=config["training"]["physics_lambda"],
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    print("   --> Complete! Residual architecture & autograd constraints compiled.")

    # 3. Train Model
    print("\n🏋️ [Step 3/7] Training Residual PINN Architecture...")
    train_loss_history = []
    physics_loss_history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_phys_loss = 0.0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            logits = model(x_b)
            bce_loss = criterion(logits, y_b)
            phys_loss = physics_loss_fn(model, x_b, device)
            loss = bce_loss + phys_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_phys_loss += phys_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_phys_loss = total_phys_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        physics_loss_history.append(avg_phys_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"   [Epoch {epoch:03d}/{config['training']['epochs']}] Total Loss: {avg_loss:.4f} | Physics Loss: {avg_phys_loss:.4f}"
            )
    print("   --> Complete! Training finished.")

    # 4. Post-Hoc Calibration
    print("\n🌡️ [Step 4/7] Performing L-BFGS Temperature Scaling Calibration...")
    temp_scaler = TemperatureScaler().to(device)
    optimal_temp = temp_scaler.calibrate(val_loader, model, device)
    print(f"   --> Complete! Optimal Temperature Parameter (T): {optimal_temp:.4f}")

    # 5. Test Inference & Evaluation
    print("\n🧪 [Step 5/7] Running test set inference and computing metrics...")
    model.eval()
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        raw_logits = model(x_test_tensor)
        calibrated_logits = temp_scaler(raw_logits)

        raw_probs = torch.sigmoid(raw_logits).cpu().numpy().flatten()
        calibrated_probs = (
            torch.sigmoid(calibrated_logits).cpu().numpy().flatten()
        )

    metrics = evaluate_predictions(y_test, raw_probs, calibrated_probs)
    for metric_name, val in metrics.items():
        print(f"   --> {metric_name:20s}: {val:.4f}")

    # 6. Save Model Checkpoint directly inside outputs/
    print("\n💾 [Step 6/7] Saving model checkpoint...")
    checkpoint_path = os.path.join(
        config["outputs"]["output_dir"], "pinn_model.pt"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"   --> Complete! Saved checkpoint to: {checkpoint_path}")

    # 7. Generate All Diagnostic & Physics Figures directly inside outputs/
    print("\n📈 [Step 7/7] Generating all diagnostic and physical verification figures...")
    solar_flux_idx = (
        feature_names.index("solar_flux_f10_7_index")
        if "solar_flux_f10_7_index" in feature_names
        else 0
    )
    altitude_idx = (
        feature_names.index("orbit_altitude_km")
        if "orbit_altitude_km" in feature_names
        else 0
    )

    generate_all_plots(
        y_true=y_test,
        raw_probs=raw_probs,
        calibrated_probs=calibrated_probs,
        model=model,
        sample_inputs=x_test,
        feature_names=feature_names,
        train_losses=train_loss_history,
        physics_losses=physics_loss_history,
        solar_flux=x_test[:, solar_flux_idx],
        altitude=x_test[:, altitude_idx],
        device=device,
    )
    print("   --> Complete! All figures saved directly inside outputs/")
    print("\n✨ Pipeline execution finished successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run End-to-End Orbital Risk PINN Pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
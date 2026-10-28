"""CLI Training Script for Orbital Conjunction Risk PINN."""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from orbital_risk.data import prepare_data_pipeline
from orbital_risk.loss import MonotonicityPhysicsLoss
from orbital_risk.models import AstrodynamicPINN, TemperatureScaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Astrodynamic PINN Model.")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on device: {device}")

    # Load Data Pipeline
    train_loader, val_loader, test_loader, scaler, feature_names, _ = (
        prepare_data_pipeline(
            csv_path=config["data"]["csv_path"],
            batch_size=config["data"]["batch_size"],
            seed=config["data"]["seed"],
        )
    )

    # Initialize Model & Loss
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

    # Training Loop
    print("Training Residual PINN Architecture...")
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

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

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{config['training']['epochs']} | Train Loss: {total_loss/len(train_loader):.4f}")

    # Post-Hoc Calibration
    temp_scaler = TemperatureScaler().to(device)
    optimal_temp = temp_scaler.calibrate(val_loader, model, device)
    print(f"Optimal Calibration Temperature Parameter: {optimal_temp:.4f}")

    # Save Checkpoint
    os.makedirs(config["outputs"]["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["outputs"]["checkpoint_dir"], "pinn_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved successfully to {checkpoint_path}")


if __name__ == "__main__":
    main()
"""End-to-end integration tests for data loading and inference pipeline."""

import torch
from src.data import prepare_data_pipeline


def test_data_pipeline_execution() -> None:
    """Verifies DataLoaders construction and dataset feature dimensions."""
    train_loader, val_loader, test_loader, scaler, feature_names, (x_test, y_test) = (
        prepare_data_pipeline("data/telemetry.csv", batch_size=64, seed=42)
    )

    assert len(feature_names) > 0
    assert x_test.shape[1] == len(feature_names)
    assert len(y_test) == len(x_test)
"""Unit tests for PINN forward passes and Temperature Scaler."""

import torch
from src.models import AstrodynamicPINN, TemperatureScaler


def test_pinn_forward_shape() -> None:
    """Verifies output logit shape for arbitrary batch sizes and input feature dimensions."""
    input_dim = 24
    model = AstrodynamicPINN(input_dim=input_dim)
    dummy_input = torch.randn(8, input_dim)
    out = model(dummy_input)

    assert out.shape == (8, 1)


def test_temperature_scaler_forward() -> None:
    """Verifies forward scaling tensor shapes in TemperatureScaler."""
    scaler = TemperatureScaler()
    logits = torch.tensor([[2.0], [-1.0]])
    scaled_logits = scaler(logits)

    assert scaled_logits.shape == (2, 1)
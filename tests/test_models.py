"""Unit tests for PINN forward passes and Temperature Scaler."""

import torch
from orbital_risk.models import AstrodynamicPINN, TemperatureScaler

def test_pinn_forward_shape():
    model       = AstrodynamicPINN(input_dim=16)
    dummy_input = torch.randn(8, 16)
    out         = model(dummy_input)
    assert out.shape == (8, 1)

def test_temperature_scaler_forward():
    scaler        = TemperatureScaler()
    logits        = torch.tensor([[2.0], [-1.0]])
    scaled_logits = scaler(logits)
    assert scaled_logits.shape == (2, 1)
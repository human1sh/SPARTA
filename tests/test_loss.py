"""Unit tests for physics loss and autograd penalty."""

import torch
from orbital_risk.loss import MonotonicityPhysicsLoss
from orbital_risk.models import AstrodynamicPINN

def test_physics_loss_computation():
    model      = AstrodynamicPINN(input_dim=16)
    physics_fn = MonotonicityPhysicsLoss(miss_distance_idx=3, penalty_lambda=0.005)
    device     = torch.device("cpu")
    batch_x    = torch.randn(32, 16)
    loss_val   = physics_fn(model, batch_x, device)
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.ndim == 0
    assert loss_val.item() >= 0.0
"""Unit tests for physics loss and autograd penalty."""

import torch
from src.loss import MonotonicityPhysicsLoss
from src.models import AstrodynamicPINN


def test_physics_loss_computation() -> None:
    """Verifies scalar output and non-negativity of physics monotonicity penalty."""
    input_dim = 24
    model = AstrodynamicPINN(input_dim=input_dim)
    physics_fn = MonotonicityPhysicsLoss(miss_distance_idx=3, penalty_lambda=0.005)
    device = torch.device("cpu")

    batch_x = torch.randn(32, input_dim)
    loss_val = physics_fn(model, batch_x, device)

    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.ndim == 0
    assert loss_val.item() >= 0.0
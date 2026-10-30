"""Physics loss module for enforcing autograd risk monotonicity."""

import torch
import torch.nn as nn


class MonotonicityPhysicsLoss(nn.Module):
    """Enforces a soft penalty against positive gradients w.r.t. miss distance:
    
    d(P_risk) / d(r_miss) <= 0
    """

    def __init__(self, miss_distance_idx: int, penalty_lambda: float = 0.005) -> None:
        super().__init__()
        self.miss_distance_idx = miss_distance_idx
        self.penalty_lambda = penalty_lambda

    def forward(
        self, model: nn.Module, batch_x: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Computes autograd penalty for physically invalid positive risk trends."""
        batch_x_req = batch_x.clone().detach().to(device).requires_grad_(True)
        logits = model(batch_x_req)
        probs = torch.sigmoid(logits)

        grads = torch.autograd.grad(
            outputs=probs.sum(),
            inputs=batch_x_req,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        miss_grads = grads[:, self.miss_distance_idx]
        physics_penalty = torch.mean(torch.relu(miss_grads))
        return self.penalty_lambda * physics_penalty
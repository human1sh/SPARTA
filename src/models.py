"""Astrodynamic Neural Network Architecture and Post-Hoc Calibration."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ResidualBlock(nn.Module):
    """Residual Block preventing vanishing gradients in deep networks."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class AstrodynamicPINN(nn.Module):
    """High-Capacity Residual Deep Neural Network for Conjunction Risk."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)


class TemperatureScaler(nn.Module):
    """Post-hoc Temperature Scaling to minimize Expected Calibration Error (ECE)."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(
        self, val_loader: DataLoader, model: nn.Module, device: torch.device
    ) -> float:
        """Optimizes temperature parameter T via L-BFGS on validation logits."""
        model.eval()
        logits_list, labels_list = [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits_list.append(model(x_batch))
                labels_list.append(y_batch.to(device))

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()
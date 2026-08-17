import torch
from torch import nn


class SharedTokenEncoder(nn.Module):
    """Per-token shared MLP. This module does not mix information across tokens."""

    def __init__(self, model_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

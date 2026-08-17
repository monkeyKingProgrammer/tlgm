import torch
from torch import nn

from .causal_linear import CausalSequenceLinear


class GlobalMixingBlock(nn.Module):
    """Causal global mixer with no attention, no QKV, and no recurrent state."""

    def __init__(self, context_length: int, model_dim: int, feature_hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(model_dim)
        self.token_mix_1 = CausalSequenceLinear(context_length)
        self.token_mix_2 = CausalSequenceLinear(context_length)
        self.feature_norm = nn.LayerNorm(model_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(model_dim, feature_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x)
        y = self.token_mix_2(self.act(self.token_mix_1(y)))
        x = x + y
        y = self.feature_mlp(self.feature_norm(x))
        return x + y


class GlobalMixingNetwork(nn.Module):
    def __init__(self, context_length: int, model_dim: int, feature_hidden_dim: int, num_blocks: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GlobalMixingBlock(context_length, model_dim, feature_hidden_dim, dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

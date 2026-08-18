import torch
from torch import nn


class CausalSequenceLinear(nn.Module):
    """Learned lower-triangular mixing across sequence positions.

    Input and output shape: [batch, sequence, dim].
    The same causal position-mixing matrix is shared across feature channels.
    """

    def __init__(self, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.weight = nn.Parameter(torch.empty(sequence_length, sequence_length))
        self.bias = nn.Parameter(torch.zeros(sequence_length))
        mask = torch.tril(torch.ones(sequence_length, sequence_length))
        self.register_buffer("causal_mask", mask, persistent=False)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, "expected [B, N, D]"
        n = x.shape[1]
        if n > self.sequence_length:
            raise ValueError(f"sequence length {n} exceeds configured length {self.sequence_length}")
        weight = self.weight[:n, :n] * self.causal_mask[:n, :n]
        bias = self.bias[:n]
        y = torch.einsum("bjd,nj->bnd", x, weight)
        return y + bias.view(1, n, 1)

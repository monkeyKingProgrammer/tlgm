import torch
from torch import nn


class TokenPositionEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        return self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))

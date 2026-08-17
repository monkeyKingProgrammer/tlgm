import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import TLGMConfig, estimate_params
from .embeddings import TokenPositionEmbeddings
from .global_mixer import GlobalMixingNetwork
from .local_encoder import SharedTokenEncoder


class TLGMForCausalLM(nn.Module):
    def __init__(self, config: TLGMConfig):
        super().__init__()
        self.config = config
        self.embeddings = TokenPositionEmbeddings(config.vocab_size, config.context_length, config.embed_dim, config.dropout)
        self.input_projection = nn.Identity() if config.embed_dim == config.model_dim else nn.Linear(config.embed_dim, config.model_dim)
        self.shared_encoder = SharedTokenEncoder(config.model_dim, config.local_hidden_dim, config.dropout)
        self.global_mixer = GlobalMixingNetwork(
            config.context_length,
            config.model_dim,
            config.feature_hidden_dim,
            config.num_global_blocks,
            config.dropout,
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings and config.embed_dim == config.model_dim:
            self.lm_head.weight = self.embeddings.token_embedding.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        for block in self.global_mixer.blocks:
            nn.init.xavier_uniform_(block.token_mix_1.weight)
            token_gain = 1.0
            if self.config.scale_residual_projections:
                token_gain /= math.sqrt(2 * self.config.num_global_blocks)
            nn.init.xavier_uniform_(block.token_mix_2.weight, gain=token_gain)
            if self.config.scale_residual_projections:
                nn.init.normal_(
                    block.feature_mlp[3].weight,
                    mean=0.0,
                    std=self.config.initializer_range / math.sqrt(2 * self.config.num_global_blocks),
                )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        x = self.embeddings(input_ids)
        x = self.input_projection(x)
        x = self.shared_encoder(x)
        x = self.global_mixer(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    def parameter_report(self) -> dict:
        estimated = estimate_params(self.config)
        actual = sum(p.numel() for p in self.parameters())
        report = dict(estimated)
        report["actual_total"] = actual
        report["fp32_mb"] = actual * 4 / 1024**2
        report["fp16_mb"] = actual * 2 / 1024**2
        report["bf16_mb"] = actual * 2 / 1024**2
        return report

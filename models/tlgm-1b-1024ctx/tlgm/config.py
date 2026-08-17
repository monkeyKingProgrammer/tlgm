from dataclasses import asdict, dataclass


@dataclass
class TLGMConfig:
    vocab_size: int = 8192
    context_length: int = 256
    embed_dim: int = 768
    model_dim: int = 768
    num_global_blocks: int = 22
    local_hidden_dim: int = 3072
    feature_hidden_dim: int = 3072
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    scale_residual_projections: bool = True

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embed_dim": self.embed_dim,
            "model_dim": self.model_dim,
            "num_global_blocks": self.num_global_blocks,
            "local_hidden_dim": self.local_hidden_dim,
            "feature_hidden_dim": self.feature_hidden_dim,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        special_ids = (self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id)
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("special token IDs must be distinct")
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in special_ids):
            raise ValueError("special token IDs must be inside the vocabulary")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TLGMConfig":
        return cls(**data)


def estimate_params(cfg: TLGMConfig) -> dict[str, int]:
    token_embedding = cfg.vocab_size * cfg.embed_dim
    position_embedding = cfg.context_length * cfg.embed_dim
    input_projection = 0 if cfg.embed_dim == cfg.model_dim else cfg.embed_dim * cfg.model_dim + cfg.model_dim
    shared_encoder = (
        cfg.model_dim * cfg.local_hidden_dim
        + cfg.local_hidden_dim
        + cfg.local_hidden_dim * cfg.model_dim
        + cfg.model_dim
        + 2 * cfg.model_dim
    )
    token_mixer_per_block = 2 * cfg.context_length * cfg.context_length + 2 * cfg.context_length
    feature_mixer_per_block = (
        cfg.model_dim * cfg.feature_hidden_dim
        + cfg.feature_hidden_dim
        + cfg.feature_hidden_dim * cfg.model_dim
        + cfg.model_dim
    )
    norms_per_block = 4 * cfg.model_dim
    global_mixer = cfg.num_global_blocks * (token_mixer_per_block + feature_mixer_per_block + norms_per_block)
    final_norm = 2 * cfg.model_dim
    output_head = 0 if cfg.tie_embeddings and cfg.embed_dim == cfg.model_dim else cfg.model_dim * cfg.vocab_size
    total = token_embedding + position_embedding + input_projection + shared_encoder + global_mixer + final_norm + output_head
    inactive_upper_triangle = (
        cfg.num_global_blocks * 2 * cfg.context_length * (cfg.context_length - 1) // 2
    )
    return {
        "token_embedding": token_embedding,
        "position_embedding": position_embedding,
        "input_projection": input_projection,
        "shared_encoder": shared_encoder,
        "global_mixer": global_mixer,
        "final_norm": final_norm,
        "output_head": output_head,
        "total": total,
        "inactive_masked_sequence_values": inactive_upper_triangle,
        "forward_active_total": total - inactive_upper_triangle,
    }

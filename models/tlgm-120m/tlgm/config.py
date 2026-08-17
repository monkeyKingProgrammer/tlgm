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
    return {
        "token_embedding": token_embedding,
        "position_embedding": position_embedding,
        "input_projection": input_projection,
        "shared_encoder": shared_encoder,
        "global_mixer": global_mixer,
        "final_norm": final_norm,
        "output_head": output_head,
        "total": total,
    }

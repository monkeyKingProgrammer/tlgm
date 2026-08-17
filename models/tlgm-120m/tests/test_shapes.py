import torch

from tlgm import TLGMConfig, TLGMForCausalLM


def test_shapes():
    cfg = TLGMConfig(context_length=16, vocab_size=128, embed_dim=64, model_dim=64, num_global_blocks=2, local_hidden_dim=128, feature_hidden_dim=128)
    model = TLGMForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    out = model(input_ids)
    assert out["logits"].shape == (2, cfg.context_length, cfg.vocab_size)

import torch

from tlgm import TLGMConfig, TLGMForCausalLM


def test_forward_loss():
    cfg = TLGMConfig(context_length=16, vocab_size=128, embed_dim=64, model_dim=64, num_global_blocks=2, local_hidden_dim=128, feature_hidden_dim=128)
    model = TLGMForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    out = model(input_ids, input_ids.clone())
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])

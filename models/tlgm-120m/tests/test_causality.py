import torch

from tlgm import TLGMConfig, TLGMForCausalLM


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(0)
    cfg = TLGMConfig(context_length=12, vocab_size=128, embed_dim=64, model_dim=64, num_global_blocks=2, local_hidden_dim=128, feature_hidden_dim=128)
    model = TLGMForCausalLM(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
    b = a.clone()
    b[:, 7:] = torch.randint(0, cfg.vocab_size, (1, cfg.context_length - 7))
    with torch.no_grad():
        logits_a = model(a)["logits"][:, :7]
        logits_b = model(b)["logits"][:, :7]
    assert torch.allclose(logits_a, logits_b, atol=1e-5)

import torch

from tlgm import TLGMConfig, TLGMForCausalLM
from tlgm.generation import generate_ids


def test_generation():
    cfg = TLGMConfig(context_length=16, vocab_size=128, embed_dim=64, model_dim=64, num_global_blocks=2, local_hidden_dim=128, feature_hidden_dim=128)
    model = TLGMForCausalLM(cfg)
    input_ids = torch.tensor([[1, 5, 6]], dtype=torch.long)
    out = generate_ids(model, input_ids, max_new_tokens=4, eos_token_id=2, temperature=0.0)
    assert out.shape[1] >= input_ids.shape[1]

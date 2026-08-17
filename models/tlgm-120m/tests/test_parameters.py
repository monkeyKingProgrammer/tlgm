from tlgm import TLGMConfig, TLGMForCausalLM


def test_default_parameter_count():
    model = TLGMForCausalLM(TLGMConfig())
    params = sum(p.numel() for p in model.parameters())
    assert 118_000_000 <= params <= 122_000_000

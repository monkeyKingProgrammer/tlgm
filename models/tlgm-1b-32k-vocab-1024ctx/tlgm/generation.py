import torch
import torch.nn.functional as F


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < values[..., -1, None], torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    return logits


@torch.inference_mode()
def generate_ids(model, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
    model.eval()
    cfg = model.config
    for _ in range(max_new_tokens):
        context = input_ids[:, -cfg.context_length :]
        logits = model(context)["logits"][:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = top_k_top_p_filter(logits / temperature, top_k=top_k, top_p=top_p)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if int(next_id[0, 0]) == eos_token_id:
            break
    return input_ids

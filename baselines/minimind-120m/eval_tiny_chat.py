import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
MINIMIND_DIR = PROJECT_DIR.parent / "MiniMind"
sys.path.insert(0, str(MINIMIND_DIR))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


PROMPTS = [
    "hello",
    "What is 2+3?",
    "What is the capital of France?",
    "Explain what a cat is.",
    "What is a dog?",
    "What is a chicken?",
    "Write a 3 sentence story about a robot.",
    "Who won the private moon chess championship in 1842?",
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def last_loss(path: Path) -> str:
    if not path.exists():
        return "not available"
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    if not last:
        return "not available"
    value = last.get("eval_loss", last.get("loss"))
    return f"{value:.4f}" if isinstance(value, (int, float)) else "not available"


def load_model(checkpoint: Path, device: str):
    cfg = MiniMindConfig(hidden_size=960, num_hidden_layers=10, use_moe=False, vocab_size=6400, num_attention_heads=10, num_key_value_heads=5)
    model = MiniMindForCausalLM(cfg)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload, strict=True)
    return model.half().eval().to(device)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, checkpoint_kind: str, device: str, max_new_tokens: int) -> str:
    if checkpoint_kind == "pretrain":
        text = tokenizer.bos_token + prompt
    else:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, open_thinking=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    for stop in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(stop, 1)[0]
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def eval_checkpoint(checkpoint: Path, tokenizer, device: str, max_new_tokens: int) -> list[tuple[str, str]]:
    if not checkpoint.exists():
        return [(p, f"missing checkpoint: {checkpoint}") for p in PROMPTS]
    model = load_model(checkpoint, device)
    kind = "pretrain" if "pretrain" in checkpoint.name else "sft"
    return [(prompt, generate(model, tokenizer, prompt, kind, device, max_new_tokens)) for prompt in PROMPTS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_checkpoint", default="checkpoints/pretrain_120m_960_2b.pth")
    parser.add_argument("--sft_checkpoint", default="checkpoints/sft_120m_960_final.pth")
    parser.add_argument("--tokenizer_path", default="../MiniMind/model")
    parser.add_argument("--output", default="outputs/eval_results.md")
    parser.add_argument("--pretrain_log", default="outputs/pretrain_120m_2b_loss.jsonl")
    parser.add_argument("--sft_log", default="outputs/sft_120m_repair_loss.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(resolve(args.tokenizer_path))
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    before = eval_checkpoint(resolve(args.pretrain_checkpoint), tokenizer, args.device, args.max_new_tokens)
    after = eval_checkpoint(resolve(args.sft_checkpoint), tokenizer, args.device, args.max_new_tokens)

    with output.open("w", encoding="utf-8") as f:
        f.write("# 120M Tiny Chat Evaluation\n\n")
        f.write(f"- Final pretrain loss: {last_loss(resolve(args.pretrain_log))}\n")
        f.write(f"- Final SFT/repair loss: {last_loss(resolve(args.sft_log))}\n\n")
        f.write("## Sample Outputs Before SFT\n\n")
        for prompt, response in before:
            f.write(f"**User:** {prompt}\n\n**Assistant:** {response}\n\n")
        f.write("## Sample Outputs After SFT\n\n")
        for prompt, response in after:
            f.write(f"**User:** {prompt}\n\n**Assistant:** {response}\n\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
MINIMIND_DIR = PROJECT_DIR.parent / "MiniMind"
sys.path.insert(0, str(MINIMIND_DIR))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def clean(text: str) -> str:
    for stop in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(stop, 1)[0]
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, args) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=args.context_length).to(args.device)
    output = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
    )
    return clean(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sft_120m_960_final.pth")
    parser.add_argument("--tokenizer_path", default="../MiniMind/model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("prompts", nargs="*")
    args = parser.parse_args()

    prompts = args.prompts or [
        "hello",
        "what is 1+5",
        "what is the capital of france?",
        "explain what a cat is",
        "what is a dog",
        "what is a chicken",
        "what is a ghost",
        "write a 3 sentence story about a robot",
        "Who won the private moon chess championship in 1842?",
    ]
    tokenizer = AutoTokenizer.from_pretrained(resolve(args.tokenizer_path))
    cfg = MiniMindConfig(hidden_size=960, num_hidden_layers=10, use_moe=False, vocab_size=6400, num_attention_heads=10, num_key_value_heads=5)
    model = MiniMindForCausalLM(cfg)
    payload = torch.load(resolve(args.checkpoint), map_location=args.device)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload, strict=True)
    model = model.half().eval().to(args.device)

    for prompt in prompts:
        print(f"User: {prompt}")
        print(f"Assistant: {generate(model, tokenizer, prompt, args)}")
        print()


if __name__ == "__main__":
    main()

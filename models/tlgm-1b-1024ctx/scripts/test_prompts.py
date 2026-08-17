import argparse
import sys
from pathlib import Path

import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.config import TLGMConfig  # noqa: E402
from tlgm.generation import generate_ids  # noqa: E402
from tlgm.model import TLGMForCausalLM  # noqa: E402
from tlgm.tokenizer import load_tokenizer  # noqa: E402


def clean(text: str) -> str:
    for stop in ("<eos>", "<pad>", "<bos>", "User:"):
        text = text.split(stop, 1)[0]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_tlgm_1b_polish.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/tlgm_1b_1024ctx_sft_final.pth")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("prompts", nargs="*")
    args = parser.parse_args()

    prompts = args.prompts or [
        "hello",
        "what is 1+5",
        "what is the capital of france?",
        "explain what a cat is",
        "what is a dog",
        "what is a chicken",
    ]

    with (PROJECT_DIR / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tokenizer = load_tokenizer(PROJECT_DIR / args.tokenizer_dir)
    model = TLGMForCausalLM(TLGMConfig.from_dict(cfg["model"]))
    payload = torch.load(PROJECT_DIR / args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload, strict=True)
    model = model.to(args.device).eval()
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    for prompt in prompts:
        rendered = f"User: {prompt}\nAssistant: "
        ids = [bos_id] + tokenizer.encode(rendered).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
        out = generate_ids(model, input_ids, args.max_new_tokens, eos_id, args.temperature, args.top_k, args.top_p)
        reply = clean(tokenizer.decode(out[0, input_ids.shape[1] :].tolist()))
        print(f"User: {prompt}")
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()

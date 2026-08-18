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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_tlgm_1b_32k_50b.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/tlgm_1b_32k_pretrain50b.pth")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with (PROJECT_DIR / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tokenizer = load_tokenizer(PROJECT_DIR / args.tokenizer_dir)
    model = TLGMForCausalLM(TLGMConfig.from_dict(cfg["model"]))
    payload = torch.load(PROJECT_DIR / args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload, strict=True)
    model = model.to(args.device).eval()
    ids = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(args.prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    out = generate_ids(model, input_ids, args.max_new_tokens, tokenizer.token_to_id("<eos>"), args.temperature, args.top_k, args.top_p)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.config import TLGMConfig  # noqa: E402
from tlgm.generation import generate_ids  # noqa: E402
from tlgm.model import TLGMForCausalLM  # noqa: E402
from tlgm.tokenizer import load_tokenizer  # noqa: E402


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def load_model(args):
    with resolve(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tokenizer = load_tokenizer(resolve(args.tokenizer_dir))
    model = TLGMForCausalLM(TLGMConfig.from_dict(cfg["model"]))
    payload = torch.load(resolve(args.checkpoint), map_location=args.device)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload, strict=True)
    return model.to(args.device).eval(), tokenizer


def render_prompt(history: list[tuple[str, str]], user: str) -> str:
    parts = []
    for old_user, old_assistant in history:
        parts.append(f"User: {old_user}")
        parts.append(f"Assistant: {old_assistant}")
    parts.append(f"User: {user}")
    parts.append("Assistant: ")
    return "\n".join(parts)


def clean_reply(text: str) -> str:
    for stop in ("<eos>", "<pad>", "<bos>", "User:"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


@torch.inference_mode()
def reply(model, tokenizer, prompt: str, args) -> str:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    ids = [bos_id] + tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    output = generate_ids(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    new_ids = output[0, input_ids.shape[1] :].tolist()
    return clean_reply(tokenizer.decode(new_ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_tlgm_120m.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/tlgm_120m_sft_polished.pth")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--history_turns", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    history: list[tuple[str, str]] = []
    print("Type 'exit' or 'quit' to stop. Type '/reset' to clear history.")
    while True:
        user = input("User: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if user.lower() == "/reset":
            history = []
            print("Assistant: history cleared")
            continue
        if not user:
            continue
        prompt = render_prompt(history[-args.history_turns :], user) if args.history_turns > 0 else render_prompt([], user)
        assistant = reply(model, tokenizer, prompt, args)
        print(f"Assistant: {assistant}")
        if args.history_turns > 0:
            history.append((user, assistant))


if __name__ == "__main__":
    main()

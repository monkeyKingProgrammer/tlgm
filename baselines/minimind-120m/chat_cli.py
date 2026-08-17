import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
MINIMIND_DIR = PROJECT_DIR.parent / "MiniMind"
sys.path.insert(0, str(MINIMIND_DIR))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(resolve(args.tokenizer_path))
    cfg = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
        vocab_size=6400,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
    )
    model = MiniMindForCausalLM(cfg)
    payload = torch.load(resolve(args.checkpoint), map_location=args.device)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload, strict=True)
    model = model.half().eval().to(args.device)
    return model, tokenizer


def build_inputs(tokenizer, history, context_length: int, device: str):
    kept = history[:]
    while kept:
        prompt = tokenizer.apply_chat_template(kept, tokenize=False, add_generation_prompt=True, open_thinking=False)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length)
        if encoded["input_ids"].shape[1] < context_length:
            return encoded.to(device), kept
        kept = kept[2:] if len(kept) > 2 else kept[-1:]
    prompt = tokenizer.apply_chat_template(history[-1:], tokenize=False, add_generation_prompt=True, open_thinking=False)
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length).to(device), history[-1:]


def clean_response(text: str) -> str:
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


@torch.inference_mode()
def generate_reply(model, tokenizer, inputs, args) -> str:
    generated = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return clean_response(tokenizer.decode(new_tokens, skip_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sft_120m_960_final.pth")
    parser.add_argument("--tokenizer_path", default="../MiniMind/model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_size", type=int, default=960)
    parser.add_argument("--num_hidden_layers", type=int, default=10)
    parser.add_argument("--num_attention_heads", type=int, default=10)
    parser.add_argument("--num_key_value_heads", type=int, default=5)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--history_turns", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    history = []
    print("Type 'exit' or 'quit' to stop. Type '/reset' to clear conversation history.")
    while True:
        user = input("User: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if user.lower() == "/reset":
            history = []
            print("Assistant: Conversation history cleared.")
            continue
        if not user:
            continue
        if args.history_turns <= 0:
            history = [{"role": "user", "content": user}]
        else:
            history.append({"role": "user", "content": user})
            history = history[-args.history_turns * 2 - 1 :]
        inputs, history = build_inputs(tokenizer, history, args.context_length, args.device)
        reply = generate_reply(model, tokenizer, inputs, args)
        print(f"Assistant: {reply}")
        if args.history_turns > 0:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

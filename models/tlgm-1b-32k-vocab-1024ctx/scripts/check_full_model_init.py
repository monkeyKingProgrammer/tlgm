import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.config import TLGMConfig  # noqa: E402
from tlgm.model import TLGMForCausalLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_tlgm_1b_32k_50b.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sequence_length", type=int, default=32)
    args = parser.parse_args()

    with (PROJECT_DIR / args.config).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    model = TLGMForCausalLM(TLGMConfig.from_dict(raw_config["model"])).to(args.device).eval()
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, args.sequence_length),
        device=args.device,
    )
    amp_context = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if "cuda" in args.device
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        result = model(input_ids, input_ids.clone())
    report = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(result["loss"].detach()),
        "finite": bool(torch.isfinite(result["loss"])),
        "embedding_std": float(model.embeddings.token_embedding.weight.detach().std()),
        "sequence_length": args.sequence_length,
        "device": args.device,
    }
    print(json.dumps(report, indent=2))
    if report["parameters"] != 1_015_592_960:
        raise ValueError("Unexpected parameter count")
    if not report["finite"] or report["loss"] >= 20.0:
        raise ValueError("Initialization produced an invalid loss")


if __name__ == "__main__":
    main()

import torch


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("No CUDA GPU detected.")
        return
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    free, total = torch.cuda.mem_get_info(idx)
    print(f"GPU name: {props.name}")
    print(f"Total VRAM: {total / 1024**3:.2f} GiB")
    print(f"Free VRAM: {free / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()

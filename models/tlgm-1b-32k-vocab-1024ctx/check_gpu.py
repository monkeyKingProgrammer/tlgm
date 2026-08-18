import torch


def gib(value: int) -> float:
    return value / 1024**3


print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    print(f"GPU: {torch.cuda.get_device_name(index)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"VRAM total: {gib(total):.2f} GiB")
    print(f"VRAM free: {gib(free):.2f} GiB")

"""Explicit CUDA selection for the user's RTX 3050 training jobs."""

import torch


def training_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; requested GPU training cannot start")
    devices = [i for i in range(torch.cuda.device_count()) if "3050" in torch.cuda.get_device_name(i)]
    if not devices:
        raise RuntimeError("Requested RTX 3050 was not found among CUDA devices")
    index = devices[0]
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def gpu_details():
    device = training_device()
    index = torch.device(device).index
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": device,
        "name": properties.name,
        "vram_gb": properties.total_memory / 1024**3,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "scope": "PyTorch pretraining, IQL, recurrent PPO, SAC, QR-DQN; statistical estimators and data preparation use CPU",
    }

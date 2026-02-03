import torch


def set_torch_config(unified_memory: bool = True):
    """
    Configure PyTorch for optimal training performance.

    Optimizations for PGX G10 (128GB LPDDR5 unified memory):
    - TF32 enabled for faster matmul/conv operations
    - cuDNN benchmark for kernel auto-tuning
    - High precision matmul for balance of speed and accuracy

    Args:
        unified_memory: If True, applies optimizations for unified memory architecture
    """
    # Reproducibility
    torch.manual_seed(999)
    torch.cuda.manual_seed(999)

    # Enable TF32 for Ampere+ GPUs (19-bit mantissa, sufficient for training)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Auto-tune convolution/attention kernels for the current hardware
    # First few iterations may be slower, but subsequent ones are faster
    torch.backends.cudnn.benchmark = True

    # Use TF32 precision for float32 matmul (faster with minimal accuracy loss)
    torch.set_float32_matmul_precision("high")

    # Deterministic algorithms (disable for max performance, enable for reproducibility)
    # torch.backends.cudnn.deterministic = True  # Uncomment if reproducibility needed

    if unified_memory:
        # Additional optimizations for unified memory architecture
        # Memory allocator tuning is done via environment variable in train script
        pass

    print(
        f"PyTorch config: TF32={torch.backends.cuda.matmul.allow_tf32}, "
        f"cuDNN.benchmark={torch.backends.cudnn.benchmark}, "
        f"matmul_precision=high, unified_memory={unified_memory}"
    )

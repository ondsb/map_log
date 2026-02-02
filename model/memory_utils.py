"""
Memory utilities for training optimization.

Optimized for unified memory architecture (PGX G10 128GB LPDDR5).
"""

import gc
from typing import Optional, Dict, Any

import torch


def get_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """
    Get current GPU memory statistics.
    
    Returns dict with:
    - allocated_gb: Currently allocated memory
    - reserved_gb: Currently reserved memory (includes fragmentation)
    - peak_gb: Peak allocated memory since last reset
    - free_gb: Estimated free memory
    """
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    
    # Get total memory
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    free = total - reserved
    
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "peak_gb": round(peak, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
    }


def log_memory_stats(prefix: str = "", device: str = "cuda") -> None:
    """Print memory statistics with optional prefix."""
    stats = get_memory_stats(device)
    if stats:
        print(f"{prefix}Memory: allocated={stats['allocated_gb']:.2f}GB, "
              f"peak={stats['peak_gb']:.2f}GB, "
              f"free={stats['free_gb']:.2f}GB")


def clear_memory(device: str = "cuda") -> None:
    """
    Aggressively clear GPU memory.
    
    Use between training phases or when OOM is expected.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def optimize_memory_for_training() -> None:
    """
    Configure PyTorch memory allocator for optimal training.
    
    Should be called before model creation.
    """
    import os
    
    # Expandable segments reduce fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # For unified memory, also consider:
    # - max_split_size_mb: Limits size of cached blocks
    # - garbage_collection_threshold: When to run GC
    

def estimate_model_memory(
    n_params: int,
    batch_size: int,
    block_size: int,
    n_embd: int,
    n_layer: int,
    dtype: str = "float16",
    optimizer: str = "adamw",
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.
    
    Returns dict with memory estimates in GB.
    """
    bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype]
    
    # Model parameters
    param_memory = n_params * bytes_per_param
    
    # Gradients (same size as parameters)
    grad_memory = n_params * bytes_per_param
    
    # Optimizer states (AdamW: 2x for momentum + variance, stored in fp32)
    if optimizer == "adamw":
        optim_memory = n_params * 4 * 2  # 2 states, fp32
    else:
        optim_memory = 0
    
    # Activations (rough estimate)
    # For transformer: ~2 * batch * seq * hidden * n_layers bytes
    activation_memory = 2 * batch_size * block_size * n_embd * n_layer * bytes_per_param
    
    # Total
    total = param_memory + grad_memory + optim_memory + activation_memory
    
    return {
        "params_gb": round(param_memory / 1e9, 2),
        "grads_gb": round(grad_memory / 1e9, 2),
        "optimizer_gb": round(optim_memory / 1e9, 2),
        "activations_gb": round(activation_memory / 1e9, 2),
        "total_gb": round(total / 1e9, 2),
    }


class MemoryTracker:
    """
    Context manager to track memory usage during operations.
    
    Usage:
        with MemoryTracker("forward pass") as tracker:
            output = model(input)
        print(tracker.stats)
    """
    
    def __init__(self, name: str = "", device: str = "cuda"):
        self.name = name
        self.device = device
        self.start_allocated: float = 0
        self.end_allocated: float = 0
        self.peak_during: float = 0
    
    def __enter__(self) -> "MemoryTracker":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_allocated = torch.cuda.memory_allocated(self.device) / 1e9
        return self
    
    def __exit__(self, *args) -> None:
        if torch.cuda.is_available():
            self.end_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            self.peak_during = torch.cuda.max_memory_allocated(self.device) / 1e9
    
    @property
    def stats(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "start_gb": round(self.start_allocated, 2),
            "end_gb": round(self.end_allocated, 2),
            "peak_gb": round(self.peak_during, 2),
            "delta_gb": round(self.end_allocated - self.start_allocated, 2),
        }


def print_memory_summary(model: torch.nn.Module, config: Any) -> None:
    """Print a summary of memory usage and estimates."""
    print("\n" + "=" * 60)
    print("MEMORY SUMMARY")
    print("=" * 60)
    
    # Current usage
    stats = get_memory_stats()
    if stats:
        print(f"\nCurrent GPU Memory:")
        print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {stats['reserved_gb']:.2f} GB")
        print(f"  Total:     {stats['total_gb']:.2f} GB")
        print(f"  Free:      {stats['free_gb']:.2f} GB")
    
    # Estimates
    n_params = sum(p.numel() for p in model.parameters())
    dtype = getattr(config, 'dtype', 'float16')
    
    estimates = estimate_model_memory(
        n_params=n_params,
        batch_size=config.batch_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        dtype=dtype,
    )
    
    print(f"\nEstimated Requirements (batch_size={config.batch_size}):")
    print(f"  Parameters:   {estimates['params_gb']:.2f} GB")
    print(f"  Gradients:    {estimates['grads_gb']:.2f} GB")
    print(f"  Optimizer:    {estimates['optimizer_gb']:.2f} GB")
    print(f"  Activations:  {estimates['activations_gb']:.2f} GB")
    print(f"  TOTAL:        {estimates['total_gb']:.2f} GB")
    
    print("=" * 60 + "\n")

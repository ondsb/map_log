"""
Memory-efficient Dataset implementation for Dota2 training.

Key optimizations:
- Pre-encodes all data as contiguous tensors (avoids list-of-lists)
- Pre-pads to block_size during init (avoids per-batch padding)
- Uses int64 directly (optimal for unified memory - no per-batch dtype conversion)
- Stores targets inline (shifted by 1) to avoid recomputation

Unified Memory Architecture (PGX G10):
- CPU and GPU share 128GB LPDDR5 memory
- No pin_memory or non_blocking needed (no actual data transfer)
- Direct tensor access without copy overhead
"""

import math

import torch
from torch.utils.data import Dataset, DataLoader


class Dota2Dataset(Dataset):
    """
    Memory-efficient dataset that stores pre-encoded, pre-padded sequences.
    
    Optimized for unified memory architecture (PGX G10 128GB LPDDR5):
    - x_t: (N, block_size) int64 - input tokens (no per-batch conversion needed)
    - x_n: (N, block_size) float32 - input numbers  
    - y_t: (N, block_size) int64 - target tokens (shifted by 1)
    - y_n: (N, block_size) float32 - target numbers (shifted by 1)
    
    Using int64 directly avoids dtype conversion overhead on each __getitem__ call.
    With 128GB unified memory, the ~4x memory increase is negligible.
    """
    
    def __init__(
        self,
        data: list[list[str]],
        tokenizer,
        block_size: int,
        device: str = "cuda",
        unified_memory: bool = True,
    ):
        self.block_size = block_size
        self.device = device
        self.pad_token = tokenizer.pad
        self.num_token = tokenizer.num
        self.unified_memory = unified_memory
        
        # Encode all data once
        n_samples = len(data)
        
        # Pre-allocate tensors with optimal dtypes for unified memory
        # int64 directly - avoids per-sample dtype conversion overhead
        # Memory cost: ~4x vs int16, but negligible with 128GB unified memory
        self.x_t = torch.zeros((n_samples, block_size), dtype=torch.int64)
        self.x_n = torch.ones((n_samples, block_size), dtype=torch.float32)
        self.y_t = torch.full((n_samples, block_size), tokenizer.pad, dtype=torch.int64)
        self.y_n = torch.ones((n_samples, block_size), dtype=torch.float32)
        
        # Encode and pad in-place
        for i, seq in enumerate(data):
            seq_len = min(len(seq), block_size)
            
            # Encode tokens and numbers
            for j, token in enumerate(seq[:seq_len]):
                if self._is_number(token):
                    self.x_t[i, j] = self.num_token
                    self.x_n[i, j] = float(token)
                else:
                    self.x_t[i, j] = tokenizer.enc[token]
                    # x_n stays 1 (default)
            
            # Pad remaining with pad token
            if seq_len < block_size:
                self.x_t[i, seq_len:] = self.pad_token
            
            # Create targets (shifted by 1)
            if seq_len > 1:
                self.y_t[i, :seq_len-1] = self.x_t[i, 1:seq_len]
                self.y_n[i, :seq_len-1] = self.x_n[i, 1:seq_len]
        
        print(f"Dataset initialized: {n_samples} samples, "
              f"{self.x_t.element_size() * self.x_t.nelement() / 1e6:.1f}MB tokens, "
              f"{self.x_n.element_size() * self.x_n.nelement() / 1e6:.1f}MB numbers, "
              f"unified_memory={unified_memory}")
    
    @staticmethod
    def _is_number(s: str) -> bool:
        try:
            return math.isfinite(float(s))
        except ValueError:
            return False
    
    def __len__(self) -> int:
        return len(self.x_t)
    
    def __getitem__(self, idx: int):
        # No dtype conversion needed - already int64
        return (
            self.x_t[idx],
            self.x_n[idx],
            self.y_t[idx],
            self.y_n[idx],
        )


def create_dataloader(
    dataset: Dota2Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,  # Disabled for unified memory architecture
    drop_last: bool = True,
    ddp: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """
    Create a DataLoader optimized for the target memory architecture.
    
    For unified memory (PGX G10):
    - pin_memory=False (no benefit, adds overhead)
    - num_workers can be > 0 (no memory duplication penalty)
    """
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if ddp else None
    
    # Only set prefetch_factor if num_workers > 0
    extra_kwargs = {}
    if num_workers > 0:
        extra_kwargs['prefetch_factor'] = prefetch_factor or 2
        extra_kwargs['persistent_workers'] = True
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **extra_kwargs,
    )


class InfiniteDataLoader:
    """
    Wrapper that creates an infinite iterator over a DataLoader.
    Resets automatically when exhausted.
    
    Optimized for unified memory: simplified .to() calls since
    CPU and GPU share the same memory space.
    """
    
    def __init__(self, dataloader: DataLoader, unified_memory: bool = True):
        self.dataloader = dataloader
        self.unified_memory = unified_memory
        self._iterator = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            try:
                return next(self._iterator)
            except StopIteration:
                raise RuntimeError("DataLoader is empty!")
    
    def get_batch(self, device: str = "cuda"):
        """
        Get next batch and move to device.
        
        For unified memory architecture:
        - Tensors are already accessible by both CPU and GPU
        - Simple .to(device) is sufficient (no pin_memory/non_blocking needed)
        - No dtype conversion needed (stored as int64)
        """
        x_t, x_n, y_t, y_n = next(self)
        
        if self.unified_memory:
            # Unified memory: direct device placement, minimal overhead
            return (
                x_t.to(device),
                x_n.to(device),
                y_t.to(device),
                y_n.to(device),
            )
        else:
            # Discrete GPU: use non_blocking for async transfer
            return (
                x_t.to(device, non_blocking=True),
                x_n.to(device, non_blocking=True),
                y_t.to(device, non_blocking=True),
                y_n.to(device, non_blocking=True),
            )

"""
Memory-efficient Dataset implementation for Dota2 training.

Key optimizations:
- Pre-encodes all data as contiguous tensors (avoids list-of-lists)
- Pre-pads to block_size during init (avoids per-batch padding)
- Uses int16 for tokens (vocab < 32k) to reduce memory
- Stores targets inline (shifted by 1) to avoid recomputation
"""

import math

import torch
from torch.utils.data import Dataset, DataLoader


class Dota2Dataset(Dataset):
    """
    Memory-efficient dataset that stores pre-encoded, pre-padded sequences.
    
    Memory layout:
    - x_t: (N, block_size) int16 - input tokens
    - x_n: (N, block_size) float32 - input numbers  
    - y_t: (N, block_size) int16 - target tokens (shifted by 1)
    - y_n: (N, block_size) float32 - target numbers (shifted by 1)
    """
    
    def __init__(
        self,
        data: list[list[str]],
        tokenizer,
        block_size: int,
        device: str = "cuda",
    ):
        self.block_size = block_size
        self.device = device
        self.pad_token = tokenizer.pad
        self.num_token = tokenizer.num
        
        # Encode all data once
        n_samples = len(data)
        
        # Pre-allocate tensors with optimal dtypes
        # int16 for tokens (vocab typically < 32k)
        self.x_t = torch.zeros((n_samples, block_size), dtype=torch.int16)
        self.x_n = torch.ones((n_samples, block_size), dtype=torch.float32)
        self.y_t = torch.full((n_samples, block_size), tokenizer.pad, dtype=torch.int16)
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
              f"{self.x_n.element_size() * self.x_n.nelement() / 1e6:.1f}MB numbers")
    
    @staticmethod
    def _is_number(s: str) -> bool:
        try:
            return math.isfinite(float(s))
        except ValueError:
            return False
    
    def __len__(self) -> int:
        return len(self.x_t)
    
    def __getitem__(self, idx: int):
        return (
            self.x_t[idx].to(torch.int64),
            self.x_n[idx],
            self.y_t[idx].to(torch.int64),
            self.y_n[idx],
        )


def create_dataloader(
    dataset: Dota2Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory=True, 
    drop_last=True,
    ddp: bool = False, # Add this flag
) -> DataLoader:
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if ddp else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None), # Shuffle only if not using DDP sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory, 
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


class InfiniteDataLoader:
    """
    Wrapper that creates an infinite iterator over a DataLoader.
    Resets automatically when exhausted.
    """
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
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
            x_t, x_n, y_t, y_n = next(self)
            # Move to GPU first, then cast
            return (
                x_t.to(device, non_blocking=True).to(torch.int64),
                x_n.to(device, non_blocking=True),
                y_t.to(device, non_blocking=True).to(torch.int64),
                y_n.to(device, non_blocking=True),
            )

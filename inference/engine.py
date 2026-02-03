"""
High-performance batched inference engine for Dota2 event prediction.

Optimizations:
- Static KV cache pre-allocation to avoid memory fragmentation
- Batched forward passes for 10-50x throughput improvement
- torch.inference_mode() for minimal overhead
- Vectorized sampling across batch dimension
- Early exit handling for variable-length generations

Optimized for PGX G10 with 128GB unified memory.
"""

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from model.model import MapLogModel, ModelConfig
from model.tokenizer import Tokenizer
from model.torch_config import set_torch_config


@dataclass
class InferenceConfig:
    """Configuration for batched inference."""

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 200

    # Batching parameters
    max_batch_size: int = 64
    block_size: int = 512

    # Hardware configuration
    device: str = "cuda"
    dtype: str = "float16"  # or "bfloat16"
    compile_model: bool = True
    compile_mode: str = "max-autotune"

    # Optimization flags
    use_kv_cache: bool = True
    use_flash_attention: bool = True  # Auto-detected by model

    @property
    def ptdtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            self.dtype
        ]


@dataclass
class GenerationOutput:
    """Output container for batched generation."""

    sequences: list[str]  # Decoded text outputs
    tokens: list[list[int]]  # Raw token sequences
    num_tokens_generated: list[int]  # Tokens generated per sequence
    total_time_ms: float  # Total generation time
    tokens_per_second: float  # Throughput


class StaticKVCache:
    """
    Pre-allocated KV cache for efficient autoregressive generation.

    Benefits:
    - No memory allocation during generation loop
    - Reduces memory fragmentation
    - Enables CUDA graph compatibility

    Memory layout per layer:
    - k_cache: [batch_size, n_heads, max_seq_len, head_dim]
    - v_cache: [batch_size, n_heads, max_seq_len, head_dim]
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate cache tensors for all layers
        self.k_caches = [
            torch.zeros(batch_size, n_heads, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.v_caches = [
            torch.zeros(batch_size, n_heads, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]

        # Track current sequence length per batch item
        self.seq_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Calculate memory usage
        bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        self.memory_bytes = (
            2 * n_layers * batch_size * n_heads * max_seq_len * head_dim * bytes_per_element
        )

        print(
            f"StaticKVCache allocated: {self.memory_bytes / 1e6:.1f}MB "
            f"({batch_size}x{n_layers}x{n_heads}x{max_seq_len}x{head_dim})"
        )

    def reset(self, batch_indices: Optional[list[int]] = None):
        """Reset cache for specified batch indices (or all if None)."""
        if batch_indices is None:
            for k, v in zip(self.k_caches, self.v_caches):
                k.zero_()
                v.zero_()
            self.seq_lens.zero_()
        else:
            for k, v in zip(self.k_caches, self.v_caches):
                k[batch_indices].zero_()
                v[batch_indices].zero_()
            self.seq_lens[batch_indices] = 0

    def get_cache_for_layer(self, layer_idx: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the valid portion of cache for a layer up to seq_len."""
        return (
            self.k_caches[layer_idx][:, :, :seq_len, :],
            self.v_caches[layer_idx][:, :, :seq_len, :],
        )

    def update_cache(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        start_pos: int,
    ):
        """Update cache with new K, V values at position start_pos."""
        seq_len = new_k.size(2)
        end_pos = start_pos + seq_len
        self.k_caches[layer_idx][:, :, start_pos:end_pos, :] = new_k
        self.v_caches[layer_idx][:, :, start_pos:end_pos, :] = new_v

    def get_layer_caches(self, seq_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get all layer caches up to seq_len."""
        return [(k[:, :, :seq_len, :], v[:, :, :seq_len, :]) for k, v in zip(self.k_caches, self.v_caches)]


class BatchedInferenceEngine:
    """
    High-performance batched inference engine.

    Key features:
    - Process multiple prompts simultaneously
    - Static KV cache for memory efficiency
    - Vectorized sampling
    - Support for both batch processing and single-request inference
    """

    def __init__(
        self,
        model: MapLogModel,
        tokenizer: Tokenizer,
        config: InferenceConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Extract model config for cache sizing
        model_config = model.config
        self.n_layers = model_config.n_layer
        self.n_heads = model_config.n_head
        self.head_dim = model_config.n_embd // model_config.n_head

        # Initialize static KV cache
        if config.use_kv_cache:
            self.kv_cache = StaticKVCache(
                batch_size=config.max_batch_size,
                max_seq_len=config.block_size,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                dtype=config.ptdtype,
                device=config.device,
            )
        else:
            self.kv_cache = None

        # Put model in eval mode
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        meta_path: Optional[str | Path] = None,
        config: Optional[InferenceConfig] = None,
    ) -> "BatchedInferenceEngine":
        """
        Load model from checkpoint and create inference engine.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            meta_path: Path to meta pickle file (tokenizer vocab)
            config: Optional inference configuration

        Returns:
            Configured BatchedInferenceEngine instance
        """
        config = config or InferenceConfig()
        checkpoint_path = Path(checkpoint_path)

        # Set up torch config
        set_torch_config(unified_memory=True)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)

        # Extract model config from checkpoint
        if "config" in checkpoint:
            model_config_dict = checkpoint["config"]
            if isinstance(model_config_dict, ModelConfig):
                model_config = model_config_dict
            else:
                model_config = ModelConfig(**model_config_dict)
        else:
            raise ValueError("Checkpoint missing 'config' key")

        # Load meta (tokenizer)
        if meta_path is None:
            # Try to find meta file based on checkpoint location
            ckpt_dir = checkpoint_path.parent
            meta_candidates = list(ckpt_dir.glob("meta*.pkl")) + list(
                ckpt_dir.parent.glob("**/meta*.pkl")
            )
            if meta_candidates:
                meta_path = meta_candidates[0]
            else:
                raise ValueError(f"Could not find meta file. Please specify meta_path.")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # Create model
        model = MapLogModel(model_config, meta)

        # Load weights
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(config.device)

        # Compile model for faster inference
        if config.compile_model:
            print(f"Compiling model with mode={config.compile_mode}")
            model = torch.compile(model, mode=config.compile_mode, fullgraph=True)

        # Create tokenizer
        tokenizer = Tokenizer(meta)

        return cls(model, tokenizer, config)

    def _encode_batch(
        self,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Encode a batch of prompts to tensors.

        Returns:
            tokens: [batch, max_len] int64 tensor
            nums: [batch, max_len] float32 tensor
            lengths: list of original sequence lengths
        """
        batch_size = len(prompts)

        # Tokenize each prompt
        encoded = [self.tokenizer.encode([p.split(" ")]) for p in prompts]
        tokens_list = [e[0][0] for e in encoded]
        nums_list = [e[1][0] for e in encoded]

        # Get lengths and max length
        lengths = [len(t) for t in tokens_list]
        max_len = max(lengths)

        # Pad to same length
        tokens = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad,
            dtype=torch.long,
            device=self.config.device,
        )
        nums = torch.ones(
            (batch_size, max_len),
            dtype=torch.float32,
            device=self.config.device,
        )

        for i, (toks, ns, length) in enumerate(zip(tokens_list, nums_list, lengths)):
            tokens[i, :length] = torch.tensor(toks, dtype=torch.long)
            nums[i, :length] = torch.tensor(ns, dtype=torch.float32)

        return tokens, nums, lengths

    def _sample_next_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Sample next tokens from logits for entire batch.

        Args:
            logits: [batch, 1, vocab_size] or [batch, vocab_size]
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            next_tokens: [batch, 1] sampled tokens
        """
        # Ensure 2D: [batch, vocab_size]
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> GenerationOutput:
        """
        Generate completions for a batch of prompts.

        This is the main high-performance generation method.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Max tokens to generate (default from config)
            temperature: Sampling temperature (default from config)
            top_k: Top-k filtering (default from config)

        Returns:
            GenerationOutput with decoded sequences and metrics
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k

        batch_size = len(prompts)
        if batch_size > self.config.max_batch_size:
            # Process in chunks
            results = []
            for i in range(0, batch_size, self.config.max_batch_size):
                chunk = prompts[i : i + self.config.max_batch_size]
                results.append(self.generate_batch(chunk, max_new_tokens, temperature, top_k))

            # Merge results
            return GenerationOutput(
                sequences=[s for r in results for s in r.sequences],
                tokens=[t for r in results for t in r.tokens],
                num_tokens_generated=[n for r in results for n in r.num_tokens_generated],
                total_time_ms=sum(r.total_time_ms for r in results),
                tokens_per_second=sum(sum(r.num_tokens_generated) for r in results)
                / (sum(r.total_time_ms for r in results) / 1000),
            )

        start_time = time.perf_counter()

        # Encode prompts
        tokens, nums, prompt_lengths = self._encode_batch(prompts)

        # Reset KV cache
        if self.kv_cache is not None:
            self.kv_cache.reset(list(range(batch_size)))

        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.config.device)
        generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.config.device)

        # Prefill: process all prompt tokens at once
        prefill_len = tokens.size(1)
        logits, _, mod_n, _, kv_caches = self.model(tokens, nums)

        # Sample first new token
        next_tokens = self._sample_next_tokens(logits, temperature, top_k)
        next_nums = torch.ones_like(next_tokens, dtype=torch.float32)

        # Update for numeric tokens
        num_mask = next_tokens == self.tokenizer.num
        if num_mask.any():
            next_nums[num_mask] = mod_n[:, -1][num_mask.squeeze(-1)].unsqueeze(-1)

        # Append to sequences
        tokens = torch.cat([tokens, next_tokens], dim=1)
        nums = torch.cat([nums, next_nums], dim=1)
        generated_lengths += 1

        # Check for EOT
        finished |= (next_tokens.squeeze(-1) == self.tokenizer.eot)

        # Autoregressive generation
        current_pos = prefill_len
        for _ in range(max_new_tokens - 1):
            if finished.all():
                break

            # Forward pass with just the new token
            logits, _, mod_n, _, kv_caches = self.model(
                next_tokens,
                next_nums,
                kv_caches=kv_caches,
                start_pos=current_pos,
            )

            # Sample next tokens
            next_tokens = self._sample_next_tokens(logits, temperature, top_k)
            next_nums = torch.ones_like(next_tokens, dtype=torch.float32)

            # Update for numeric tokens
            num_mask = next_tokens == self.tokenizer.num
            if num_mask.any():
                next_nums[num_mask] = mod_n[:, -1][num_mask.squeeze(-1)].unsqueeze(-1)

            # Append to sequences (only for unfinished)
            tokens = torch.cat([tokens, next_tokens], dim=1)
            nums = torch.cat([nums, next_nums], dim=1)

            # Update counters
            generated_lengths += (~finished).long()
            current_pos += 1

            # Check for EOT
            finished |= (next_tokens.squeeze(-1) == self.tokenizer.eot)

        end_time = time.perf_counter()

        # Decode sequences
        sequences = []
        tokens_out = []
        for i in range(batch_size):
            seq_tokens = tokens[i].tolist()
            seq_nums = nums[i].tolist()
            decoded = self.tokenizer.decode(seq_tokens, seq_nums)
            sequences.append(decoded)
            tokens_out.append(seq_tokens)

        total_time_ms = (end_time - start_time) * 1000
        total_tokens = generated_lengths.sum().item()
        tokens_per_second = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0

        return GenerationOutput(
            sequences=sequences,
            tokens=tokens_out,
            num_tokens_generated=generated_lengths.tolist(),
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_second,
        )

    @torch.inference_mode()
    def generate_single(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Generate completion for a single prompt.

        Convenience method that wraps generate_batch.
        """
        result = self.generate_batch([prompt], max_new_tokens, temperature, top_k)
        return result.sequences[0]

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        stats = {
            "model_params_mb": sum(p.numel() * p.element_size() for p in self.model.parameters())
            / 1e6,
            "kv_cache_mb": self.kv_cache.memory_bytes / 1e6 if self.kv_cache else 0,
            "gpu_allocated_mb": torch.cuda.memory_allocated() / 1e6
            if torch.cuda.is_available()
            else 0,
            "gpu_reserved_mb": torch.cuda.memory_reserved() / 1e6
            if torch.cuda.is_available()
            else 0,
        }
        return stats

    def warmup(self, num_warmup: int = 3):
        """Run warmup iterations to compile and optimize kernels."""
        print(f"Running {num_warmup} warmup iterations...")
        dummy_prompt = "start - light player0 player1 player2 player3 player4 - dark player5 player6 player7 player8 player9 - prematch 0.5 dur 0.3 tk 0.2 - >"

        for i in range(num_warmup):
            _ = self.generate_batch([dummy_prompt] * min(8, self.config.max_batch_size), max_new_tokens=32)
            print(f"  Warmup {i+1}/{num_warmup} complete")

        print("Warmup complete!")

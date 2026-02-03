"""
Transformer building blocks with architecture optimizations.

Optimizations included:
- RoPE (Rotary Position Embeddings): Better extrapolation, relative positions
- SwiGLU activation: Better training dynamics than GELU
- Fourier numeric embeddings: Better continuous value representation
- Flash Attention: Memory-efficient attention (PyTorch 2.0+)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (bias=False is slightly more efficient)."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input=x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-5,
        )


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm (no mean subtraction, no bias).
    Used in LLaMA, Mistral, and other modern architectures.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Advantages over learned absolute positions:
    - Better length extrapolation
    - Encodes relative positions naturally
    - No additional parameters to learn
    - Compatible with Flash Attention

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin values
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._seq_len_cached: int = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if sequence length changed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin for rotary embeddings.

        Args:
            x: Input tensor (used for device/dtype)
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) tensors of shape [seq_len, dim]
        """
        self._update_cache(seq_len, x.device, x.dtype)
        return (
            self._cos_cached[:seq_len],
            self._sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]

    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Reshape cos/sin for broadcasting: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FourierNumericEmbedding(nn.Module):
    """
    Fourier feature embedding for continuous numeric values.

    Better than simple scalar multiplication for representing continuous values.
    Uses sinusoidal features at multiple frequencies to create a rich embedding.

    Reference: "Fourier Features Let Networks Learn High Frequency Functions"
    https://arxiv.org/abs/2006.10739
    """

    def __init__(self, n_embd: int, num_frequencies: int = 32, learnable: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies

        if learnable:
            # Learnable frequency scales (initialized with log-uniform distribution)
            self.log_freqs = nn.Parameter(
                torch.linspace(0, math.log(100), num_frequencies)
            )
        else:
            # Fixed frequencies (log-spaced)
            log_freqs = torch.linspace(0, math.log(100), num_frequencies)
            self.register_buffer("log_freqs", log_freqs)

        # Project Fourier features to embedding dimension
        self.proj = nn.Linear(num_frequencies * 2, n_embd)

    @property
    def freqs(self) -> torch.Tensor:
        return torch.exp(self.log_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Numeric values tensor [batch, seq_len]

        Returns:
            Embeddings tensor [batch, seq_len, n_embd]
        """
        # x: [B, T] -> [B, T, num_frequencies]
        angles = x.unsqueeze(-1) * self.freqs

        # Concatenate sin and cos features
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        # Project to embedding dimension
        return self.proj(features)


class NumericEmbedding(nn.Module):
    """
    Learnable numeric embedding with optional Fourier features.

    Combines:
    1. Direct linear projection (captures linear relationships)
    2. Fourier features (captures periodic/non-linear patterns)

    This replaces the simple scalar multiplication approach which
    scales ALL embedding dimensions equally and has limited expressivity.
    """

    def __init__(
        self,
        n_embd: int,
        use_fourier: bool = True,
        num_frequencies: int = 32,
    ):
        super().__init__()
        self.use_fourier = use_fourier

        if use_fourier:
            self.fourier = FourierNumericEmbedding(n_embd, num_frequencies)
        else:
            # Simple learnable projection
            self.proj = nn.Linear(1, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Numeric values tensor [batch, seq_len]

        Returns:
            Embeddings tensor [batch, seq_len, n_embd]
        """
        if self.use_fourier:
            return self.fourier(x)
        else:
            return self.proj(x.unsqueeze(-1))


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE support.

    Features:
    - Flash Attention for memory efficiency (PyTorch 2.0+)
    - Optional RoPE for better position encoding
    - Efficient QKV projection
    """

    def __init__(
        self, config: "GPTConfig", rotary_emb: Optional[RotaryEmbedding] = None
    ) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.rotary_emb = rotary_emb

        # Check for Flash Attention support
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            print(
                "WARNING: Flash Attention not available. Using manual attention (slower)."
            )
            # Fallback causal mask
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV cache for efficient inference.

        Args:
            x: Input tensor [B, T, C]
            kv_cache: Optional tuple of (k_cache, v_cache) each [B, n_head, cache_len, head_dim]
            start_pos: Starting position in sequence (for RoPE and cache indexing)

        Returns:
            output: Attention output [B, T, C]
            new_kv_cache: Updated (k_cache, v_cache) if caching, else None
        """
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE if available (with correct position offset)
        if self.rotary_emb is not None:
            # For cached inference, we need positions from start_pos to start_pos + T
            seq_len = start_pos + T
            cos, sin = self.rotary_emb(x, seq_len)
            # Only use the positions for current tokens
            cos = cos[start_pos:seq_len]
            sin = sin[start_pos:seq_len]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache for incremental decoding
        new_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Concatenate new K, V with cached values
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            new_kv_cache = (k, v)
        elif not self.training:
            # During inference without cache, still return cache for next iteration
            new_kv_cache = (k, v)

        # Get full sequence length for attention
        S = k.size(2)  # Full key/value sequence length

        # Compute attention
        if self.flash:
            # Flash Attention (memory efficient, fused kernel)
            # Note: When using cache, we need is_causal=False and manual masking
            if kv_cache is not None:
                # Incremental decoding: query attends to all cached + current keys
                # No causal mask needed since we're only generating one token at a time
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,  # No dropout during inference
                    is_causal=False,  # Full attention to cached context
                )
            else:
                # Prefill or training: use causal attention
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
        else:
            # Manual attention (fallback)
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale

            if kv_cache is None:
                # Causal mask for prefill/training
                att = att.masked_fill(
                    self.causal_mask[:, :, :T, :S] == 0, float("-inf")
                )

            att = F.softmax(att, dim=-1)
            if self.training:
                att = self.attn_dropout(att)
            y = att @ v

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv_cache


class MLP(nn.Module):
    """
    Standard MLP with GELU activation.

    Architecture: Linear -> GELU -> Linear -> Dropout
    Expansion factor: 4x
    """

    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP (Swish-Gated Linear Unit).

    Better training dynamics than standard GELU MLP.
    Used in LLaMA, PaLM, and other state-of-the-art models.

    Architecture:
    - gate = Swish(Linear(x))
    - hidden = Linear(x)
    - output = Linear(gate * hidden)

    Hidden dimension is 2/3 of standard 4x to match parameter count.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        # 2/3 of 4x expansion to compensate for gating (keeps param count similar)
        # Rounded to nearest multiple of 64 for efficiency
        hidden_dim = int(4 * config.n_embd * 2 / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # Round to multiple of 64

        self.w1 = nn.Linear(
            config.n_embd, hidden_dim, bias=config.bias
        )  # Gate projection
        self.w2 = nn.Linear(
            hidden_dim, config.n_embd, bias=config.bias
        )  # Output projection
        self.w3 = nn.Linear(
            config.n_embd, hidden_dim, bias=config.bias
        )  # Up projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(gate) * hidden
        gate = F.silu(self.w1(x))  # Swish activation
        hidden = self.w3(x)
        x = gate * hidden
        x = self.w2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block with pre-normalization.

    Architecture: LN -> Attention -> Residual -> LN -> MLP -> Residual

    Supports:
    - Standard MLP or SwiGLU MLP
    - LayerNorm or RMSNorm
    - RoPE via attention module
    """

    def __init__(
        self,
        config: "GPTConfig",
        rotary_emb: Optional[RotaryEmbedding] = None,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()

        # Normalization layers
        if use_rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        # Attention
        self.attn = CausalSelfAttention(config, rotary_emb=rotary_emb)

        # MLP
        if use_swiglu:
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV cache support.

        Args:
            x: Input tensor [B, T, C]
            kv_cache: Optional KV cache from previous forward pass
            start_pos: Starting position for RoPE

        Returns:
            output: Block output [B, T, C]
            new_kv_cache: Updated KV cache if caching, else None
        """
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache, start_pos)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

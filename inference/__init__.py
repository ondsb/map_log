"""
High-performance inference package for Dota2 event prediction.

Optimized for PGX G10 workstation with 128GB unified memory.

Key features:
- Batched inference for 10-50x throughput improvement
- Static KV cache to avoid memory allocation during generation
- torch.inference_mode() for minimal overhead
- Configurable batch sizes and sequence lengths

Usage:
    # Load engine from checkpoint
    from inference import BatchedInferenceEngine, InferenceConfig
    
    config = InferenceConfig(max_batch_size=64, compile_model=True)
    engine = BatchedInferenceEngine.from_checkpoint("model.pt", config=config)
    
    # Run warmup
    engine.warmup()
    
    # Generate predictions
    results = engine.generate_batch(prompts, max_new_tokens=512)
    print(f"Throughput: {results.tokens_per_second:.1f} tok/s")

CLI Usage:
    # Run benchmarks
    python -m inference.benchmark --checkpoint model.pt --compare
    
    # Batch process matches
    python -m inference.run_batch --checkpoint model.pt --test
    
    # Start API server
    python -m inference.server --checkpoint model.pt --port 8000
"""

from inference.engine import (
    BatchedInferenceEngine,
    StaticKVCache,
    InferenceConfig,
    GenerationOutput,
)

__all__ = [
    "BatchedInferenceEngine",
    "StaticKVCache",
    "InferenceConfig",
    "GenerationOutput",
]

__version__ = "0.1.0"

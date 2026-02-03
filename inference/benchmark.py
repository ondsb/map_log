"""
Benchmark suite for inference throughput testing.

Usage:
    python -m inference.benchmark --checkpoint /path/to/model.pt --meta /path/to/meta.pkl
    python -m inference.benchmark --checkpoint model.pt --batch-sizes 1,8,16,32,64 --seq-lengths 100,200,500

Features:
- Test throughput across different batch sizes
- Compare old sequential vs new batched inference
- GPU utilization monitoring
- Pretty-printed results with speedup calculations
"""

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from inference.engine import BatchedInferenceEngine, InferenceConfig


@dataclass
class BenchmarkResult:
    """Result for a single benchmark configuration."""

    batch_size: int
    num_sequences: int
    seq_length: int  # Average generated length
    total_tokens: int
    total_time_ms: float
    tokens_per_second: float
    latency_per_seq_ms: float
    gpu_memory_mb: float
    gpu_peak_mb: float


def generate_test_prompts(num_prompts: int) -> list[str]:
    """Generate realistic test prompts for benchmarking."""
    # Template based on actual Dota2 game structure
    base_prompt = (
        "start - light player0 player1 player2 player3 player4 - "
        "dark player5 player6 player7 player8 player9 - "
        "prematch {prematch} dur {dur} tk {tk} - >"
    )

    prompts = []
    for i in range(num_prompts):
        # Vary the prematch probability slightly for diversity
        prematch = 0.3 + (i % 5) * 0.1
        dur = 0.2 + (i % 4) * 0.05
        tk = 0.1 + (i % 3) * 0.05
        prompts.append(base_prompt.format(prematch=prematch, dur=dur, tk=tk))

    return prompts


def run_benchmark(
    engine: BatchedInferenceEngine,
    batch_sizes: list[int],
    num_sequences: int = 32,
    max_new_tokens: int = 256,
    warmup_runs: int = 3,
    benchmark_runs: int = 5,
) -> list[BenchmarkResult]:
    """
    Run benchmark across different batch sizes.

    Args:
        engine: Configured BatchedInferenceEngine
        batch_sizes: List of batch sizes to test
        num_sequences: Total sequences per test
        max_new_tokens: Max tokens to generate per sequence
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of timed benchmark iterations

    Returns:
        List of BenchmarkResult for each batch size
    """
    # Generate test prompts
    prompts = generate_test_prompts(num_sequences)
    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking batch_size={batch_size}")
        print(f"{'='*60}")

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        print(f"  Running {warmup_runs} warmup iterations...")
        test_prompts = prompts[:batch_size]
        for _ in range(warmup_runs):
            _ = engine.generate_batch(test_prompts, max_new_tokens=min(64, max_new_tokens))

        # Benchmark runs
        print(f"  Running {benchmark_runs} benchmark iterations...")
        times = []
        total_tokens = []
        seq_lengths = []

        for run in range(benchmark_runs):
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            output = engine.generate_batch(prompts, max_new_tokens=max_new_tokens)
            end = time.perf_counter()

            times.append((end - start) * 1000)
            total_tokens.append(sum(output.num_tokens_generated))
            seq_lengths.append(sum(output.num_tokens_generated) / len(prompts))

        # Calculate statistics
        avg_time_ms = sum(times) / len(times)
        avg_tokens = sum(total_tokens) / len(total_tokens)
        avg_seq_len = sum(seq_lengths) / len(seq_lengths)
        tokens_per_sec = avg_tokens / (avg_time_ms / 1000)
        latency_per_seq = avg_time_ms / num_sequences

        # Memory stats
        gpu_memory_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

        result = BenchmarkResult(
            batch_size=batch_size,
            num_sequences=num_sequences,
            seq_length=int(avg_seq_len),
            total_tokens=int(avg_tokens),
            total_time_ms=avg_time_ms,
            tokens_per_second=tokens_per_sec,
            latency_per_seq_ms=latency_per_seq,
            gpu_memory_mb=gpu_memory_mb,
            gpu_peak_mb=gpu_peak_mb,
        )

        results.append(result)
        print(f"  Results: {tokens_per_sec:.1f} tok/s, {latency_per_seq:.1f}ms/seq")

    return results


def print_results(results: list[BenchmarkResult], baseline_result: Optional[BenchmarkResult] = None):
    """Pretty-print benchmark results with optional speedup comparison."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(
        f"{'Batch':<8} {'Seqs':<6} {'Tokens':<8} {'Time(ms)':<10} "
        f"{'Tok/s':<10} {'Lat/seq':<10} {'Mem(MB)':<10} {'Peak(MB)':<10}"
        + (" {'Speedup':<8}" if baseline_result else "")
    )
    print("-" * 80)

    for r in results:
        speedup = ""
        if baseline_result:
            speedup_val = r.tokens_per_second / baseline_result.tokens_per_second
            speedup = f"{speedup_val:.2f}x"

        print(
            f"{r.batch_size:<8} {r.num_sequences:<6} {r.total_tokens:<8} "
            f"{r.total_time_ms:<10.1f} {r.tokens_per_second:<10.1f} "
            f"{r.latency_per_seq_ms:<10.1f} {r.gpu_memory_mb:<10.1f} "
            f"{r.gpu_peak_mb:<10.1f}"
            + (f" {speedup}" if baseline_result else "")
        )

    print("=" * 80)

    # Summary
    if len(results) > 1:
        best = max(results, key=lambda r: r.tokens_per_second)
        print(f"\nBest configuration: batch_size={best.batch_size}")
        print(f"  Throughput: {best.tokens_per_second:.1f} tokens/second")
        print(f"  Latency: {best.latency_per_seq_ms:.1f} ms/sequence")
        print(f"  Memory: {best.gpu_peak_mb:.1f} MB peak")

        if baseline_result:
            speedup = best.tokens_per_second / baseline_result.tokens_per_second
            print(f"  Speedup vs baseline: {speedup:.1f}x")


def compare_sequential_vs_batched(
    engine: BatchedInferenceEngine,
    num_sequences: int = 16,
    max_new_tokens: int = 128,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Compare sequential (batch_size=1) vs batched inference.

    Returns:
        Tuple of (sequential_result, batched_result)
    """
    prompts = generate_test_prompts(num_sequences)

    # Sequential: process one at a time
    print("\nSequential inference (batch_size=1)...")
    gc.collect()
    torch.cuda.empty_cache()

    start = time.perf_counter()
    seq_tokens = 0
    for prompt in prompts:
        output = engine.generate_batch([prompt], max_new_tokens=max_new_tokens)
        seq_tokens += output.num_tokens_generated[0]
    seq_time = (time.perf_counter() - start) * 1000

    sequential_result = BenchmarkResult(
        batch_size=1,
        num_sequences=num_sequences,
        seq_length=seq_tokens // num_sequences,
        total_tokens=seq_tokens,
        total_time_ms=seq_time,
        tokens_per_second=seq_tokens / (seq_time / 1000),
        latency_per_seq_ms=seq_time / num_sequences,
        gpu_memory_mb=torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
        gpu_peak_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
    )

    # Batched: all at once
    print(f"Batched inference (batch_size={num_sequences})...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    output = engine.generate_batch(prompts, max_new_tokens=max_new_tokens)
    batch_time = (time.perf_counter() - start) * 1000
    batch_tokens = sum(output.num_tokens_generated)

    batched_result = BenchmarkResult(
        batch_size=num_sequences,
        num_sequences=num_sequences,
        seq_length=batch_tokens // num_sequences,
        total_tokens=batch_tokens,
        total_time_ms=batch_time,
        tokens_per_second=batch_tokens / (batch_time / 1000),
        latency_per_seq_ms=batch_time / num_sequences,
        gpu_memory_mb=torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
        gpu_peak_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("SEQUENTIAL vs BATCHED COMPARISON")
    print("=" * 60)
    print(f"Sequential: {sequential_result.tokens_per_second:.1f} tok/s")
    print(f"Batched:    {batched_result.tokens_per_second:.1f} tok/s")
    speedup = batched_result.tokens_per_second / sequential_result.tokens_per_second
    print(f"Speedup:    {speedup:.1f}x")
    print("=" * 60)

    return sequential_result, batched_result


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference throughput")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--meta", type=str, default=None, help="Path to meta pickle file")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,16,32,64",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument("--num-sequences", type=int, default=32, help="Total sequences per test")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--compare", action="store_true", help="Compare sequential vs batched")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--compile", action="store_true", default=True, help="Compile model")

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Create inference config
    config = InferenceConfig(
        max_batch_size=max(batch_sizes),
        max_new_tokens=args.max_tokens,
        device=args.device,
        compile_model=args.compile,
    )

    # Load engine
    print(f"Loading model from {args.checkpoint}")
    engine = BatchedInferenceEngine.from_checkpoint(
        args.checkpoint,
        meta_path=args.meta,
        config=config,
    )

    # Print engine info
    print(f"\nEngine configuration:")
    mem_stats = engine.get_memory_stats()
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.1f} MB")

    # Warmup
    engine.warmup(num_warmup=args.warmup)

    # Run comparison if requested
    if args.compare:
        seq_result, batch_result = compare_sequential_vs_batched(
            engine,
            num_sequences=min(args.num_sequences, 16),
            max_new_tokens=min(args.max_tokens, 128),
        )

    # Run full benchmark
    print("\nRunning full benchmark suite...")
    results = run_benchmark(
        engine,
        batch_sizes=batch_sizes,
        num_sequences=args.num_sequences,
        max_new_tokens=args.max_tokens,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
    )

    # Print results
    baseline = results[0] if batch_sizes[0] == 1 else None
    print_results(results, baseline_result=baseline)


if __name__ == "__main__":
    main()

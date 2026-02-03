"""
CLI script for batched inference on Dota2 matches.

Usage:
    # Process all validation matches
    python -m inference.run_batch --checkpoint model.pt --input val_matches.json --output runs/

    # Quick test mode (3 samples, single match)
    python -m inference.run_batch --checkpoint model.pt --test

    # High-throughput mode
    python -m inference.run_batch --checkpoint model.pt --batch-size 64 --num-samples 1000

Features:
- High-throughput batched generation (10-50x faster than sequential)
- Progress bar with ETA
- Resume capability (skip already processed matches)
- JSON output compatible with existing analysis pipeline (level1_runs.py)
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from inference.engine import BatchedInferenceEngine, InferenceConfig
from model.scaler import Scaler, ScalerPars


# Scalers for converting prematch probability
pr_scaler = Scaler(ScalerPars(-3, 3, 0, 1))


def prepare_prompts(text: str, num_samples: int, prematch_override: Optional[float] = None) -> list[str]:
    """
    Prepare prompts for a single match.

    Creates num_samples prompts with varying prematch probabilities:
    - First half: biased towards light win (prematch=3.0)
    - Second half: biased towards dark win (prematch=-3.0)

    Args:
        text: Raw match text from JSON
        num_samples: Number of samples to generate
        prematch_override: If set, use this prematch value for all samples

    Returns:
        List of prompt strings
    """
    # Parse the prompt structure
    parts = text.split(">")[0].split(" - ")

    # Extract original prematch probability
    prematch_data = parts[3].split(" ")
    original_prematch = pr_scaler.inverse_transform(float(prematch_data[2]))

    # Calculate sample distribution based on prematch probability
    lw_samples = round(original_prematch * num_samples)
    dw_samples = num_samples - lw_samples

    prompts = []

    # Light-win biased prompts
    for _ in range(lw_samples):
        prompt_parts = parts.copy()
        prematch_tokens = prompt_parts[3].split(" ")
        prematch_tokens[2] = "3.0"  # Bias towards light win
        prompt_parts[3] = " ".join(prematch_tokens)
        prompts.append(" - ".join(prompt_parts) + ">")

    # Dark-win biased prompts
    for _ in range(dw_samples):
        prompt_parts = parts.copy()
        prematch_tokens = prompt_parts[3].split(" ")
        prematch_tokens[2] = "-3.0"  # Bias towards dark win
        prompt_parts[3] = " ".join(prematch_tokens)
        prompts.append(" - ".join(prompt_parts) + ">")

    return prompts


def process_match(
    engine: BatchedInferenceEngine,
    match_id: str,
    text: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    batch_size: int,
) -> dict:
    """
    Process a single match with batched inference.

    Args:
        engine: Configured inference engine
        match_id: Match identifier
        text: Raw match text
        num_samples: Number of samples to generate
        max_new_tokens: Max tokens per sample
        temperature: Sampling temperature
        top_k: Top-k filtering
        batch_size: Batch size for inference

    Returns:
        Dict with match_id -> list of generated sequences
    """
    # Prepare all prompts for this match
    prompts = prepare_prompts(text, num_samples)

    # Process in batches
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        result = engine.generate_batch(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        all_outputs.extend(result.sequences)

    return {match_id: all_outputs}


def run_batch_inference(
    engine: BatchedInferenceEngine,
    matches: dict,
    output_dir: Path,
    num_samples: int = 200,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 200,
    batch_size: int = 64,
    resume: bool = True,
):
    """
    Run batched inference on all matches.

    Args:
        engine: Configured inference engine
        matches: Dict of match_id -> match text
        output_dir: Directory for output JSON files
        num_samples: Samples per match
        max_new_tokens: Max tokens per sample
        temperature: Sampling temperature
        top_k: Top-k filtering
        batch_size: Batch size for inference
        resume: Skip already processed matches
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing processed matches
    if resume:
        processed = set()
        for f in output_dir.glob("run_*.json"):
            match_id = f.stem.replace("run_", "")
            processed.add(match_id)
        print(f"Found {len(processed)} already processed matches")
        matches = {k: v for k, v in matches.items() if k not in processed}

    if not matches:
        print("No matches to process!")
        return

    print(f"\nProcessing {len(matches)} matches with {num_samples} samples each")
    print(f"Batch size: {batch_size}, Max tokens: {max_new_tokens}")
    print(f"Output directory: {output_dir}")

    total_samples = len(matches) * num_samples
    total_start = time.time()

    # Process each match
    for match_id, text in tqdm(matches.items(), desc="Processing matches"):
        match_start = time.time()

        result = process_match(
            engine,
            match_id,
            text,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            batch_size,
        )

        # Save to JSON (compatible with level1_runs.py)
        output_path = output_dir / f"run_{match_id}.json"
        with open(output_path, "w") as f:
            json.dump(result, f)

        match_time = time.time() - match_start
        samples_per_sec = num_samples / match_time
        tqdm.write(f"  {match_id}: {samples_per_sec:.1f} samples/sec")

    total_time = time.time() - total_start
    print(f"\nCompleted {len(matches)} matches in {total_time/60:.1f} minutes")
    print(f"Average: {total_samples/total_time:.1f} samples/sec")


def run_test(engine: BatchedInferenceEngine):
    """Quick test mode with minimal samples."""
    print("\n" + "=" * 60)
    print("RUNNING TEST MODE")
    print("=" * 60)

    test_prompt = (
        "start - light player0 player1 player2 player3 player4 - "
        "dark player5 player6 player7 player8 player9 - "
        "prematch 0.5 dur 0.3 tk 0.2 - >"
    )

    print(f"\nTest prompt:\n{test_prompt}\n")

    # Generate a few samples
    print("Generating 3 test samples...")
    result = engine.generate_batch(
        [test_prompt] * 3,
        max_new_tokens=256,
        temperature=0.8,
        top_k=200,
    )

    print(f"\nGeneration stats:")
    print(f"  Total time: {result.total_time_ms:.1f}ms")
    print(f"  Tokens/sec: {result.tokens_per_second:.1f}")
    print(f"  Tokens generated: {result.num_tokens_generated}")

    print(f"\nSample outputs:")
    for i, seq in enumerate(result.sequences[:3]):
        print(f"\n--- Sample {i+1} ---")
        # Print first 500 chars
        print(seq[:500] + "..." if len(seq) > 500 else seq)

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batched inference for Dota2 matches")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--meta", type=str, default=None, help="Path to meta pickle file")
    parser.add_argument("--input", type=str, default=None, help="Path to matches JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=200, help="Samples per match")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per sample")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=200, help="Top-k filtering")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip processed matches")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--no-compile", action="store_true", help="Disable model compilation")
    parser.add_argument("--test", action="store_true", help="Run quick test mode")

    args = parser.parse_args()

    # Create inference config
    config = InferenceConfig(
        max_batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        compile_model=not args.no_compile,
        block_size=512,
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
    engine.warmup(num_warmup=3)

    # Run test mode if requested
    if args.test:
        run_test(engine)
        return

    # Load matches
    if args.input is None:
        # Try to use default path from config
        from config.train_dota2 import data_version

        default_path = Path(f"data/dota2/maps_{data_version}.json")
        if default_path.exists():
            args.input = str(default_path)
        else:
            raise ValueError("No input file specified and default not found. Use --input")

    print(f"Loading matches from {args.input}")
    with open(args.input, "r") as f:
        matches = json.load(f)

    # Set output directory
    if args.output is None:
        from config.train_dota2 import data_version

        args.output = f"out/dota2_{data_version}/matches_runs/"

    # Run batch inference
    run_batch_inference(
        engine=engine,
        matches=matches,
        output_dir=Path(args.output),
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

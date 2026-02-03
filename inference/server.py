"""
Minimal REST API server for interactive inference.

Usage:
    # Start server
    python -m inference.server --checkpoint model.pt --port 8000

    # Query single prediction
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"prompt": "start - light ... - >", "num_samples": 1}'

    # Batch prediction
    curl -X POST http://localhost:8000/batch \
        -H "Content-Type: application/json" \
        -d '{"prompts": ["prompt1", "prompt2"], "max_new_tokens": 512}'

Endpoints:
    POST /predict  - Single or small batch prediction
    POST /batch    - High-throughput batch prediction
    GET  /health   - Health check
    GET  /stats    - Engine statistics
"""

import argparse
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from inference.engine import BatchedInferenceEngine, InferenceConfig

# Global engine instance (initialized at startup)
engine: Optional[BatchedInferenceEngine] = None
stats = {
    "requests": 0,
    "total_tokens_generated": 0,
    "total_time_ms": 0,
    "startup_time": None,
}


# Request/Response models
class PredictRequest(BaseModel):
    """Request for single/small batch prediction."""

    prompt: str = Field(..., description="Input prompt string")
    num_samples: int = Field(default=1, ge=1, le=16, description="Number of samples to generate")
    max_new_tokens: int = Field(default=512, ge=1, le=2048, description="Max tokens to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=200, ge=1, le=1000, description="Top-k filtering")


class PredictResponse(BaseModel):
    """Response for single prediction."""

    sequences: list[str]
    num_tokens_generated: list[int]
    elapsed_ms: float
    tokens_per_second: float


class BatchRequest(BaseModel):
    """Request for batch prediction."""

    prompts: list[str] = Field(..., min_length=1, max_length=128, description="List of prompts")
    max_new_tokens: int = Field(default=512, ge=1, le=2048, description="Max tokens to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=200, ge=1, le=1000, description="Top-k filtering")


class BatchResponse(BaseModel):
    """Response for batch prediction."""

    sequences: list[str]
    num_tokens_generated: list[int]
    total_tokens: int
    elapsed_ms: float
    tokens_per_second: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    engine_loaded: bool
    gpu_available: bool
    model_params_mb: float


class StatsResponse(BaseModel):
    """Engine statistics response."""

    requests: int
    total_tokens_generated: int
    avg_tokens_per_second: float
    uptime_seconds: float
    memory_stats: dict


# Server configuration
server_config = {
    "checkpoint": None,
    "meta_path": None,
    "device": "cuda",
    "max_batch_size": 64,
    "compile_model": True,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine on startup."""
    global engine, stats

    print("Initializing inference engine...")

    config = InferenceConfig(
        max_batch_size=server_config["max_batch_size"],
        device=server_config["device"],
        compile_model=server_config["compile_model"],
    )

    engine = BatchedInferenceEngine.from_checkpoint(
        server_config["checkpoint"],
        meta_path=server_config["meta_path"],
        config=config,
    )

    # Warmup
    engine.warmup(num_warmup=3)

    stats["startup_time"] = time.time()
    print("Engine ready!")

    yield

    # Cleanup
    print("Shutting down...")


# Create FastAPI app (only if FastAPI is available)
if HAS_FASTAPI:
    app = FastAPI(
        title="Dota2 Event Predictor",
        description="High-performance inference API for Dota2 event prediction",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        import torch

        mem_stats = engine.get_memory_stats() if engine else {}

        return HealthResponse(
            status="healthy" if engine else "initializing",
            engine_loaded=engine is not None,
            gpu_available=torch.cuda.is_available(),
            model_params_mb=mem_stats.get("model_params_mb", 0),
        )

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """Get engine statistics."""
        uptime = time.time() - stats["startup_time"] if stats["startup_time"] else 0
        avg_tps = (
            stats["total_tokens_generated"] / (stats["total_time_ms"] / 1000)
            if stats["total_time_ms"] > 0
            else 0
        )

        return StatsResponse(
            requests=stats["requests"],
            total_tokens_generated=stats["total_tokens_generated"],
            avg_tokens_per_second=avg_tps,
            uptime_seconds=uptime,
            memory_stats=engine.get_memory_stats() if engine else {},
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """Generate predictions for a single prompt."""
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # Create prompts
        prompts = [request.prompt] * request.num_samples

        # Generate
        result = engine.generate_batch(
            prompts,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

        # Update stats
        stats["requests"] += 1
        stats["total_tokens_generated"] += sum(result.num_tokens_generated)
        stats["total_time_ms"] += result.total_time_ms

        return PredictResponse(
            sequences=result.sequences,
            num_tokens_generated=result.num_tokens_generated,
            elapsed_ms=result.total_time_ms,
            tokens_per_second=result.tokens_per_second,
        )

    @app.post("/batch", response_model=BatchResponse)
    async def batch_predict(request: BatchRequest):
        """Generate predictions for multiple prompts."""
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # Generate
        result = engine.generate_batch(
            request.prompts,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

        # Update stats
        stats["requests"] += 1
        stats["total_tokens_generated"] += sum(result.num_tokens_generated)
        stats["total_time_ms"] += result.total_time_ms

        return BatchResponse(
            sequences=result.sequences,
            num_tokens_generated=result.num_tokens_generated,
            total_tokens=sum(result.num_tokens_generated),
            elapsed_ms=result.total_time_ms,
            tokens_per_second=result.tokens_per_second,
        )


def main():
    if not HAS_FASTAPI:
        print("ERROR: FastAPI not installed. Install with: pip install fastapi uvicorn")
        print("Or add to requirements.txt: fastapi>=0.109.0 uvicorn>=0.27.0")
        return

    parser = argparse.ArgumentParser(description="Inference API server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--meta", type=str, default=None, help="Path to meta pickle file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max-batch-size", type=int, default=64, help="Max batch size")
    parser.add_argument("--no-compile", action="store_true", help="Disable model compilation")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    # Update server config
    server_config["checkpoint"] = args.checkpoint
    server_config["meta_path"] = args.meta
    server_config["device"] = args.device
    server_config["max_batch_size"] = args.max_batch_size
    server_config["compile_model"] = not args.no_compile

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Max batch size: {args.max_batch_size}")

    uvicorn.run(
        "inference.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()

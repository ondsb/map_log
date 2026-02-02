import datetime
from pathlib import Path

# Training mode: scratch | resume
init_from = "scratch"
ckpt_path = None  # Set to resume from specific checkpoint

# DDP settings
backend = "nccl"

# Device configuration
device = "cuda"
device_type = "cuda"
do_compile = True

# Logging - MLflow
mlflow_tracking_uri = None  # Set to your hosted server URI
mlflow_log = True

# Dataset
dataset = "dota2"
data_version = "v0"

# ============================================================================
# MODEL ARCHITECTURE - Phase 1 Baseline (4.8M params)
# ============================================================================
n_layer = 8
n_head = 8
n_embd = 256
block_size = 512
dropout = 0.05
bias = False

# ============================================================================
# TRAINING CONFIGURATION
# Optimized for PGX G10 with 128GB LPDDR5 unified memory
# ============================================================================
# Batch size: Increased for unified memory (no CPU/GPU transfer overhead)
# With 128GB shared memory, we can use larger batches without OOM
batch_size = 128  # Increased from 32 (4x)
gradient_accumulation_steps = 1  # Reduced from 2 (larger batch compensates)
# effective batch size = batch_size * gradient_accumulation_steps = 128
# tokens per iter = effective batch size * block_size = 128 * 512 = 65,536

# Learning rate schedule
# Increased learning rate slightly to account for larger batch size
# Following linear scaling rule: lr_new = lr_old * (batch_new / batch_old)
learning_rate = 1e-3  # Increased from 5e-4 (2x for 4x batch = sqrt scaling)
max_iters = 100000
warmup_iters = 200  # Increased warmup for larger batch
decay_lr = True
lr_decay_iters = int(0.9 * max_iters)
min_lr = 1e-5

# Optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Hardware-specific configuration
# PGX G10 (Orin-based): ~137 TFLOPS FP16, 275 TFLOPS INT8
# A100: 312 TFLOPS BF16
# H100: 989 TFLOPS FP16 Tensor Core
hardware_peak_tflops = 137  # For MFU calculation

# Evaluation and checkpointing
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

# ============================================================================
# ARCHITECTURE OPTIMIZATIONS (Phase 3)
# These features improve training quality and sometimes speed
# ============================================================================
use_rope = True  # Rotary Position Embeddings (better extrapolation)
use_swiglu = True  # SwiGLU MLP (better training dynamics than GELU)
use_rmsnorm = False  # RMSNorm (slightly faster, but LayerNorm is more stable)
use_fourier_num = True  # Fourier features for numeric embedding
num_frequencies = 32  # Number of Fourier frequencies for numeric values

# ============================================================================
# OUTPUT PATHS
# ============================================================================
model_id = f"ckpt_{n_layer}_{n_head}_{n_embd}_{block_size}_{max_iters}"
meta_id = f"meta_num_{data_version}.pkl"
phase = "optimized"  # Changed from "baseline" to "optimized"

out_dir = f"ckpt/{dataset}_{data_version}_{phase}"

# ============================================================================
# DERIVED CONFIGURATION (do not modify)
# ============================================================================
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str, type(None)))]
config = {k: globals()[k] for k in config_keys}

# Experiment identification
experiment_name = "dota2_transformer"
run_name = f"optimized_{n_layer}_{n_head}_{n_embd}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

model_args = dict(
    device=device,
    device_type=device_type,
    model_id=model_id,
    out_dir=Path(out_dir),
    block_size=block_size,
    vocab_size=None,  # Loaded from tokenizer metadata
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    # Architecture optimizations
    use_rope=use_rope,
    use_swiglu=use_swiglu,
    use_rmsnorm=use_rmsnorm,
    use_fourier_num=use_fourier_num,
    num_frequencies=num_frequencies,
    # Training config
    batch_size=batch_size,
    warmup_iters=warmup_iters,
    eval_iters=eval_iters,
    eval_interval=eval_interval,
    learning_rate=learning_rate,
    decay_lr=decay_lr,
    lr_decay_iters=lr_decay_iters,
    min_lr=min_lr,
    weight_decay=weight_decay,
    beta1=beta1,
    beta2=beta2,
    max_iters=max_iters,
    gradient_accumulation_steps=gradient_accumulation_steps,
    grad_clip=grad_clip,
    mlflow_log=mlflow_log,
    log_interval=log_interval,
    always_save_checkpoint=always_save_checkpoint,
    data_version=data_version,
    # Hardware config
    hardware_peak_tflops=hardware_peak_tflops,
)

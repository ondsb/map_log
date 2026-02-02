"""
Phase 1: Baseline Configuration (4.8M parameters)
-------------------------------------------------
Purpose: Establish ground truth metrics on ThinkStation Blackwell
- Profile memory footprint
- Measure tokens/sec throughput
- Validate training convergence matches previous runs

Model: 8 layers, 8 heads, 256 embedding dim
Expected memory: ~8-12GB peak
"""

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
n_layer = 16
n_head = 16
n_embd = 1024
block_size = 1024
dropout = 0.05
bias = False

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
batch_size = 128  # Reduced for 5.6GB GPU
gradient_accumulation_steps = 2  # Increased to maintain effective batch size
# Effective batch size = 8 * 16 = 128 sequences
# Tokens per iter = 128 * 1024 = 131,072 tokens

# Learning rate schedule
learning_rate = 5e-4
max_iters = 100000
warmup_iters = 100
decay_lr = True
lr_decay_iters = int(0.9 * max_iters)
min_lr = 1e-5

# Optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Evaluation and checkpointing
eval_interval = 5
eval_iters = 1  # Reduced for faster/lighter evaluations on small GPU
log_interval = 1
always_save_checkpoint = True

# ============================================================================
# OUTPUT PATHS
# ============================================================================
model_id = f"ckpt_{n_layer}_{n_head}_{n_embd}_{block_size}_{max_iters}"
meta_id = f"meta_num_{data_version}.pkl"
phase = "baseline"

out_dir = f"ckpt/{dataset}_{data_version}_{phase}"

# ============================================================================
# DERIVED CONFIGURATION (do not modify)
# ============================================================================
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str, type(None)))]
config = {k: globals()[k] for k in config_keys}

# Experiment identification
experiment_name = "dota2_transformer"
run_name = f"baseline_{n_layer}_{n_head}_{n_embd}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
)

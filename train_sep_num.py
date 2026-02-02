import os
from contextlib import nullcontext

import torch
import mlflow

# Configure PyTorch memory allocator for better fragmentation handling
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from config.train_dota2_baseline import *
from dota2.data.data_train import get_train_val_meta
from dota2.data.dataset import Dota2Dataset, create_dataloader, InfiniteDataLoader
from model.GPT import GPT, GPTConfig
from model.torch_config import set_torch_config

os.makedirs(out_dir, exist_ok=True)

# dtype and autocast context
dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

mlflow_tracking_uri = "arn:aws:sagemaker:eu-west-1:192663853223:mlflow-app/app-WZTYJFWD56IS"

# mlflow
if mlflow_log:
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(
        {
            "phase": phase,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "block_size": block_size,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_iters": max_iters,
            "dtype": dtype,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }
    )

# Load raw data
train_data, val_data, meta = get_train_val_meta()

# Set up torch optimizations before model creation
set_torch_config()

# Create model BEFORE loading data to GPU
gpt_conf = GPTConfig(**model_args)
model = GPT(gpt_conf, meta)

if mlflow_log:
    mlflow.log_params(
        {
            "num_params": model.get_num_params(),
            "num_params_millions": round(model.get_num_params() / 1e6, 2),
        }
    )

# Load checkpoint if resuming
if init_from == "resume":
    iter_num, best_val_loss = model.load_ckpt(ckpt_path)
else:
    iter_num, best_val_loss = 0, 1e9

# Move model to GPU
model.to(device)

# Memory baseline (before compile)
torch.cuda.reset_peak_memory_stats()
if mlflow_log:
    mlflow.log_metric("memory/model_loaded_gb", torch.cuda.memory_allocated() / 1e9, step=0)

# Compile model with memory-efficient settings
if do_compile:
    model: GPT = torch.compile(model, mode="reduce-overhead")

# Log memory after compile
if mlflow_log:
    mlflow.log_metric("memory/after_compile_gb", torch.cuda.memory_allocated() / 1e9, step=0)

# Create efficient datasets with pre-encoded, pre-padded tensors
# This avoids per-batch NumPy allocations during training
print("Creating memory-efficient datasets...")
train_dataset = Dota2Dataset(
    data=train_data,
    tokenizer=model.generator.tokenizer,
    block_size=block_size,
    device=device,
)

val_dataset = Dota2Dataset(
    data=val_data,
    tokenizer=model.generator.tokenizer,
    block_size=block_size,
    device=device,
)


# Create DataLoaders with pinned memory for fast GPU transfer
train_loader = InfiniteDataLoader(
    create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Main process to avoid memory duplication
        pin_memory=True,
        drop_last=True,
    )
)

val_loader = InfiniteDataLoader(
    create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
)

# Log final memory state before training
if mlflow_log:
    mlflow.log_metric("memory/initial_gb", torch.cuda.memory_allocated() / 1e9, step=0)

# Initialize optimizer
model.init_optimizer()

# Train using the memory-efficient DataLoader-based method
model.do_train_dataloader(
    train_loader=train_loader,
    val_loader=val_loader,
    ddp=False,
    master_process=True,
    iter_num=iter_num,
    best_val_loss=best_val_loss,
    ctx=ctx,
    dtype=dtype,
)

if mlflow_log:
    mlflow.log_metric("memory/final_peak_gb", torch.cuda.max_memory_allocated() / 1e9)
    mlflow.end_run()

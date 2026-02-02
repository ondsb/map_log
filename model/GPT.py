import inspect
import math
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import psutil

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import mlflow

from model.modules import Block, LayerNorm, RMSNorm, RotaryEmbedding, NumericEmbedding
from model.tokenizer import Tokenizer


@dataclass
class GPTConfig:
    device: str = "cuda"
    device_type: str = "cuda"
    do_compile: bool = True
    dtype: str = "float16"
    model_id: str = "mod_0"
    out_dir: Path = Path("")
    data_version: str = ""

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    
    # Architecture optimizations (Phase 3)
    use_rope: bool = True  # Rotary Position Embeddings
    use_swiglu: bool = True  # SwiGLU activation in MLP
    use_rmsnorm: bool = False  # RMSNorm instead of LayerNorm
    use_fourier_num: bool = True  # Fourier features for numeric embedding
    num_frequencies: int = 32  # Number of Fourier frequencies

    batch_size: int = 64
    warmup_iters: int = 100
    eval_iters: int = 20
    eval_interval: int = 100
    always_save_checkpoint: bool = False

    learning_rate: float = 1e-4
    decay_lr: bool = True
    lr_decay_iters: int = 1000
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    max_iters: int = 1000
    gradient_accumulation_steps: int = 1
    grad_clip: float = 1.0

    mlflow_log: bool = False
    log_interval: int = 10

    heroes_light_pos = [4, 6, 8, 10, 12]
    heroes_dark_pos = [17, 19, 21, 23, 25]

    @property
    def _dtype(self) -> str:
        return (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )

    @property
    def ptdtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self._dtype]


class DataTN:
    data_txt: list
    data_num: list


class GenState(Enum):
    ANY = auto()
    HEROES_LIGHT = auto()
    HEROES_DARK = auto()


class Generator:
    state: GenState
    event_side: str
    masks: dict

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.state = GenState.ANY
        self.tokenizer = tokenizer

    def update_teams(
        self,
        hero_tokens_light: list[int],
        hero_tokens_dark: list[int],
    ) -> None:
        heroes_light = hero_tokens_light + [self.tokenizer.enc["team"]]
        heroes_dark = hero_tokens_dark + [self.tokenizer.enc["team"]]
        self.masks = {
            GenState.HEROES_LIGHT: [
                False if x in heroes_light else True for x in self.tokenizer.dec.keys()
            ],
            GenState.HEROES_DARK: [
                False if x in heroes_dark else True for x in self.tokenizer.dec.keys()
            ],
        }

    def update_state(self, cur_token: int) -> None:
        match cur_token:
            case self.tokenizer.side_light:
                self.event_side = "light"
                self.state = GenState.HEROES_LIGHT
            case self.tokenizer.side_dark:
                self.event_side = "dark"
                self.state = GenState.HEROES_DARK
            case self.tokenizer.kill:
                if self.event_side == "light":
                    self.state = GenState.HEROES_DARK
                else:
                    self.state = GenState.HEROES_LIGHT
            case _:
                self.state = GenState.ANY

    @property
    def is_mask_state(self) -> bool:
        return self.state in [GenState.HEROES_LIGHT, GenState.HEROES_DARK]

    @property
    def state_mask(self) -> torch.Tensor:
        return torch.tensor([[self.masks[self.state]]], device="cuda:0")


class GPT(nn.Module):
    """
    GPT Language Model with architecture optimizations.
    
    Features:
    - Dual-head output: Token prediction + Numeric regression
    - Optional RoPE (Rotary Position Embeddings)
    - Optional SwiGLU MLP activation
    - Optional RMSNorm instead of LayerNorm
    - Fourier numeric embeddings for continuous values
    """
    optimizer: torch.optim.Optimizer
    require_backward_grad_sync: bool

    def __init__(self, config, meta):
        super().__init__()
        self.generator = Generator(Tokenizer(meta))
        self.config = config
        self.ctx = torch.amp.autocast(
            device_type=config.device_type,
            dtype=config.ptdtype,
        )

        self.data_train: DataTN = DataTN()
        self.data_val: DataTN = DataTN()
        
        # Initialize RoPE if enabled (shared across all attention layers)
        rotary_emb = None
        if getattr(config, 'use_rope', True):
            head_dim = config.n_embd // config.n_head
            rotary_emb = RotaryEmbedding(head_dim, config.block_size)
            print(f"Using RoPE with head_dim={head_dim}")
        
        # Initialize numeric embedding if using Fourier features
        self.num_embed = None
        if getattr(config, 'use_fourier_num', True):
            num_frequencies = getattr(config, 'num_frequencies', 32)
            self.num_embed = NumericEmbedding(
                config.n_embd, 
                use_fourier=True, 
                num_frequencies=num_frequencies
            )
            print(f"Using Fourier numeric embedding with {num_frequencies} frequencies")
        
        # Get architecture options
        use_swiglu = getattr(config, 'use_swiglu', True)
        use_rmsnorm = getattr(config, 'use_rmsnorm', False)
        
        # Build transformer with optional position embeddings
        transformer_dict = dict(
            wte=nn.Embedding(meta["vocab_size"], config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                Block(config, rotary_emb=rotary_emb, use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm) 
                for _ in range(config.n_layer)
            ]),
        )
        
        # Add position embeddings only if not using RoPE
        if rotary_emb is None:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        
        # Add final normalization
        if use_rmsnorm:
            transformer_dict['ln_f'] = RMSNorm(config.n_embd)
        else:
            transformer_dict['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)
        
        self.transformer = nn.ModuleDict(transformer_dict)
        
        # Output heads
        self.lm_head = nn.Linear(config.n_embd, meta["vocab_size"], bias=False)
        self.num_head = nn.Linear(config.n_embd, 1)

        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Track if using RoPE (for forward pass)
        self.use_rope = rotary_emb is not None

        # apply special scaled init to residual projections
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w2.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print(f"Model initialized: {self.get_num_params()/1e6:.2f}M params, "
              f"RoPE={self.use_rope}, SwiGLU={use_swiglu}, RMSNorm={use_rmsnorm}")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_optimizer(self):
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings do decay, all biases and layernorms don't.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        d_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nd_params = [p for n, p in param_dict.items() if p.dim() < 2]

        num_d_params = sum(p.numel() for p in d_params)
        num_nd_params = sum(p.numel() for p in nd_params)

        print(f"{len(d_params)} decayed tensors, {num_d_params:,} parameters")
        print(f"{len(nd_params)} non-decayed tensors, {num_nd_params:,} parameters")

        optim_groups = [
            {"params": d_params, "weight_decay": self.config.weight_decay},
            {"params": nd_params, "weight_decay": 0.0},
        ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")

        extra_args = dict(fused=True) if use_fused else dict()
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            **extra_args,
        )

    def load_ckpt(
        self,
        ckpt_path: Path,
    ) -> tuple:
        print(f"load ckpt from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cuda")
        state_dict = checkpoint["model"]

        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        self.load_state_dict(state_dict)

        self.eval()  # todo: mby cant do for train load ckpt
        self.to(self.config.device)

        return checkpoint["iter_num"], checkpoint["best_val_loss"]

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> str:
        """
        encode -> generate -> decode wrapper
        """
        prompt_t, prompt_n = self.generator.tokenizer.encode([prompt.split(" ")])

        x_t = torch.tensor(prompt_t[0], dtype=torch.long, device=self.config.device)[
            None, ...
        ]
        x_n = torch.tensor(prompt_n[0], dtype=torch.float, device=self.config.device)[
            None, ...
        ]
        y_t, y_n = self._generate(x_t, x_n, max_new_tokens, temperature, top_k)

        return self.generator.tokenizer.decode(y_t[0], y_n[0])

    @torch.no_grad()
    def _generate(
        self,
        idx_t,
        idx_n,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> tuple:
        """
        Take a conditioning sequence of indices {idx} (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        self.update_heroes(idx_t)

        for _ in range(max_new_tokens):
            idx_t = idx_t[:, -self.config.block_size :]
            idx_n = idx_n[:, -self.config.block_size :]

            logits_t, loss_t, mod_n, loss_n = self(idx_t, idx_n)
            idx_t_next = self.next_token_from_logits(logits_t, temperature, top_k)

            if idx_t_next.item() == self.generator.tokenizer.eot:
                break

            # todo: optimise ~ idx_n only for num values & dont do else?
            if idx_t_next.item() == self.generator.tokenizer.num:
                idx_n_next = mod_n[:, 0]
            else:
                idx_n_next = torch.tensor([[1]], device=self.device_type)

            idx_t = torch.cat((idx_t, idx_t_next), dim=1)
            idx_n = torch.cat((idx_n, idx_n_next), dim=1)

            self.generator.update_state(idx_t_next.item())

        return idx_t.tolist(), idx_n.tolist()

    def update_heroes(self, out_t):
        # hero_tokens_light = [
        #     x.item()
        #     for i, x in enumerate(out_t[0])
        #     if i in self.config.heroes_light_pos
        # ]
        # hero_tokens_dark = [
        #     x.item() for i, x in enumerate(out_t[0]) if i in self.config.heroes_dark_pos
        # ]
        hero_tokens_light = [
            self.generator.tokenizer.enc[x] for x in [f"player{y}" for y in range(5)]
        ]
        hero_tokens_dark = [
            self.generator.tokenizer.enc[x]
            for x in [f"player{y}" for y in range(5, 10)]
        ]
        self.generator.update_teams(hero_tokens_light, hero_tokens_dark)

    def next_token_from_logits(
        self,
        logits_t: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        if self.generator.is_mask_state:
            logits_t = logits_t.masked_fill_(self.generator.state_mask, -float("Inf"))

        logits_t = logits_t[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits_t, min(top_k, logits_t.size(-1)))
            logits_t[logits_t < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits_t, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def forward(self, x_t, x_n, y_t=None, y_n=None):
        """
        Forward pass with dual-head output (tokens + numbers).
        
        Args:
            x_t: Token indices [batch, seq_len]
            x_n: Numeric values [batch, seq_len] (1.0 for non-numeric tokens)
            y_t: Target token indices (optional, for training)
            y_n: Target numeric values (optional, for training)
            
        Returns:
            logits_t: Token logits [batch, seq_len, vocab_size]
            loss_t: Token cross-entropy loss (None if no targets)
            mod_n: Numeric predictions [batch, seq_len]
            loss_n: Numeric MSE loss (None if no targets)
        """
        b, t = x_t.size()
        if t > self.config.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds block size {self.config.block_size}"
            )

        # Token embeddings
        tok_emb = self.transformer.wte(x_t)  # shape (b, t, n_embd)
        
        # Numeric embeddings - use Fourier features if available, else scalar multiply
        if self.num_embed is not None:
            # Fourier numeric embedding (additive, more expressive)
            num_emb = self.num_embed(x_n)  # shape (b, t, n_embd)
            tok_emb = tok_emb + num_emb
        else:
            # Legacy: scalar multiplication (all dims scaled equally)
            tok_emb = tok_emb * x_n.unsqueeze(2)

        # Position embeddings (only if not using RoPE)
        if not self.use_rope:
            pos = torch.arange(0, t, dtype=torch.long, device=x_t.device)
            pos_emb = self.transformer.wpe(pos)
            tok_emb = tok_emb + pos_emb

        # Transformer forward
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Calculate loss if given targets
        if y_t is not None:
            logits_t = self.lm_head(x)
            loss_t = F.cross_entropy(
                logits_t.view(-1, logits_t.size(-1)),
                y_t.view(-1),
                ignore_index=self.generator.tokenizer.pad,
            )

            mod_n = self.num_head(x).squeeze(-1)
            # Mask loss to only numeric positions (where y_n != 1)
            mask = (y_n != 1).float()
            masked_pred = mod_n * mask
            masked_target = y_n * mask
            # Avoid division by zero
            num_numeric = mask.sum().clamp(min=1)
            loss_n = F.mse_loss(masked_pred, masked_target, reduction='sum') / num_numeric
        else:
            # Inference: only compute for last position
            logits_t = self.lm_head(x[:, [-1], :])
            loss_t = None
            mod_n = self.num_head(x[:, [-1], :])
            loss_n = None

        return logits_t, loss_t, mod_n, loss_n

    def do_train(self, data, iter_num, best_val_loss, ddp, master_process):  # todo
        t0 = time.time()
        local_iter_num = 0
        running_mfu = 0.0
        print(f"tokens per iter: {self.tokens_per_iter:,}")

        raw_model = self.module if ddp else self

        # scaler for fp16
        scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == "float16"))

        # # todo: profiler decorator
        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #         schedule=torch.profiler.schedule(
        #             wait=5,
        #             warmup=5,
        #             active=5,
        #             repeat=3
        #         ),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        #         record_shapes=False,
        #         profile_memory=False,
        #         with_stack=False,  # incurs an additional overhead, disable if not needed
        #         with_flops=True,
        #         with_modules=False,  # only for torchscript models atm
        # ) as profiler:

        (
            self.data_train.data_txt,
            self.data_train.data_num,
        ) = self.generator.tokenizer.encode(data["train"])
        (
            self.data_val.data_txt,
            self.data_val.data_num,
        ) = self.generator.tokenizer.encode(data["val"])

        self.train()

        # todo: improve iter num and while
        x_t_train, x_n_train, y_t_train, y_n_train = self.get_batch("train")
        while True:
            lr = self._learning_rate(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if iter_num % self.config.eval_interval == 0 and master_process:
                self.log_eval(iter_num, raw_model, data, lr, running_mfu, best_val_loss)

            for micro_step in range(self.config.gradient_accumulation_steps):
                if ddp:
                    self.require_backward_grad_sync = (
                        micro_step == self.config.gradient_accumulation_steps - 1
                    )
                with self.ctx:
                    _, loss_t, _, loss_n = self(
                        x_t_train, x_n_train, y_t_train, y_n_train
                    )
                    loss = (loss_t + loss_n) / self.config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                # prefetch while model is doing forward pass on GPU
                x_t_train, x_n_train, y_t_train, y_n_train = self.get_batch("train")

            if self.config.grad_clip != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)

            scaler.step(self.optimizer)
            scaler.update()

            self.optimizer.zero_grad()

            t1 = time.time()
            if iter_num % self.config.log_interval == 0 and master_process:
                self.log_no_eval(
                    raw_model, loss, local_iter_num, t1 - t0, running_mfu, iter_num
                )
            t0 = t1

            # profiler.step()

            iter_num += 1
            local_iter_num += 1
            if iter_num > self.config.max_iters:
                break

    def do_train_dataloader(
        self,
        train_loader,
        val_loader,
        ddp,
        master_process,
        iter_num,
        best_val_loss,
        ctx,
        dtype,
    ):
        """
        Memory-efficient training loop using PyTorch DataLoaders.
        
        Key optimizations over do_train():
        - Uses pre-encoded, pre-padded tensors from DataLoader
        - No per-batch NumPy allocations
        - Proper memory cleanup with gradient zeroing
        - Prefetching batches while GPU computes
        
        Args:
            train_loader: InfiniteDataLoader for training data
            val_loader: InfiniteDataLoader for validation data  
            ddp: Whether using DistributedDataParallel
            master_process: Whether this is the master process (for logging)
            iter_num: Starting iteration number
            best_val_loss: Best validation loss so far
            ctx: Autocast context for mixed precision
            dtype: Data type string ('float16', 'bfloat16', 'float32')
        """
        t0 = time.time()
        local_iter_num = 0
        running_mfu = 0.0
        print(f"tokens per iter: {self.tokens_per_iter:,}")

        raw_model = self.module if ddp else self
        self.ctx = ctx
        
        # Store loaders for estimate_loss_dataloader
        self._train_loader = train_loader
        self._val_loader = val_loader

        # scaler for fp16
        scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

        print("starting train (dataloader mode)")
        self.train()

        # Prefetch first batch
        x_t_train, x_n_train, y_t_train, y_n_train = train_loader.get_batch(self.config.device)
        
        while True:
            lr = self._learning_rate(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if iter_num % self.config.eval_interval == 0 and master_process:
                self._log_eval_dataloader(iter_num, raw_model, lr, running_mfu, best_val_loss)
                # Re-fetch batch after eval to ensure clean memory state                
                try:
                    x_t_train, x_n_train, y_t_train, y_n_train = train_loader.get_batch(self.config.device)
                except StopIteration:
                    # Reset the iterator if it hits the end
                    train_loader.reset_iterator() 
                    x_t_train, x_n_train, y_t_train, y_n_train = train_loader.get_batch(self.config.device)

            for micro_step in range(self.config.gradient_accumulation_steps):
                if ddp:
                    self.require_backward_grad_sync = micro_step == self.config.gradient_accumulation_steps - 1
                with self.ctx:
                    _, loss_t, _, loss_n = self(x_t_train, x_n_train, y_t_train, y_n_train)
                    loss = (loss_t + loss_n) / self.config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                # Prefetch next batch while GPU is computing
                try:
                    x_t_train, x_n_train, y_t_train, y_n_train = train_loader.get_batch(self.config.device)
                except StopIteration:
                    # Reset the iterator if it hits the end
                    train_loader.reset_iterator() 
                    x_t_train, x_n_train, y_t_train, y_n_train = train_loader.get_batch(self.config.device)

            if self.config.grad_clip != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)

            scaler.step(self.optimizer)
            scaler.update()

            # Flush gradients and clear memory
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            if iter_num % self.config.log_interval == 0 and master_process:
                self.log_no_eval(raw_model, loss, local_iter_num, t1 - t0, running_mfu, iter_num)
            t0 = t1

            iter_num += 1
            local_iter_num += 1
            
            if iter_num > self.config.max_iters:
                break

    @torch.no_grad()
    def _estimate_loss_dataloader(self):
        out = {}
        self.eval()

        for split, loader in [("train", self._train_loader), ("val", self._val_loader)]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x_t, x_n, y_t, y_n = loader.get_batch(self.config.device)
                with self.ctx:
                    _, loss_t, _, loss_n = self(x_t, x_n, y_t, y_n)
                    loss = loss_t + loss_n
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.train()
        return out

    def _log_eval_dataloader(self, iter_num, raw_model, lr, running_mfu, best_val_loss):
        losses = self._estimate_loss_dataloader()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if self.config.mlflow_log:
            virtual_mem = psutil.virtual_memory()
            mlflow.log_metrics(
                {
                    "train/loss": float(losses["train"]),
                    "val/loss": float(losses["val"]),
                    "lr": lr,
                    "mfu_percent": running_mfu * 100,
                    "system/unified_mem_used_gb": virtual_mem.used / 1e9,
                    "system/unified_mem_pct": virtual_mem.percent,
                },
                step=iter_num,
            )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "model_args": self.config,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": self.config,
                }
                print(f"saving {self.config.model_id} to {self.config.out_dir}")
                ckpt_path = os.path.join(self.config.out_dir, f"{self.config.model_id}_ep{iter_num}.pt")
                torch.save(checkpoint, ckpt_path)

    def get_batch(self, split: str):
        if split == "train":
            data_txt = self.data_train.data_txt
            data_num = self.data_train.data_num
        elif split == "val":
            data_txt = self.data_val.data_txt
            data_num = self.data_val.data_num
        else:
            raise Exception("bad split")

        ix = torch.randint(len(data_txt), (self.config.batch_size,))
        block_size = self.config.block_size

        x_t = [
            F.pad(
                input=torch.from_numpy(np.array(data_txt[i]).astype(np.int64)),
                pad=(0, max(0, block_size - len(data_txt[i]))),
                mode="constant",
                value=self.generator.tokenizer.pad,
            )[: self.config.block_size]
            for i in ix
        ]
        x_t = torch.stack(x_t)

        x_n = [
            F.pad(
                input=torch.from_numpy(np.array(data_num[i]).astype(np.float32)),
                pad=(0, max(0, block_size - len(data_num[i]))),
                mode="constant",
                value=1,
            )[: self.config.block_size]
            for i in ix
        ]
        x_n = torch.stack(x_n)

        y_t = [
            F.pad(
                input=torch.from_numpy(np.array(data_txt[i][1:]).astype(np.int64)),
                pad=(0, max(0, block_size - len(data_txt[i])) + 1),
                mode="constant",
                value=self.generator.tokenizer.pad,
            )[:block_size]
            for i in ix
        ]
        y_t = torch.stack(y_t)

        y_n = [
            F.pad(
                input=torch.from_numpy(np.array(data_num[i][1:]).astype(np.float32)),
                pad=(0, max(0, block_size - len(data_num[i])) + 1),
                mode="constant",
                value=1,
            )[:block_size]
            for i in ix
        ]
        y_n = torch.stack(y_n)

        if self.device_type == "cuda":
            # pin arrays x,y, allows to move them to GPU asynchronously (non_blocking=True)
            x_t, x_n, y_t, y_n = (
                x_t.pin_memory().to(self.config.device, non_blocking=True),
                x_n.pin_memory().to(self.config.device, non_blocking=True),
                y_t.pin_memory().to(self.config.device, non_blocking=True),
                y_n.pin_memory().to(self.config.device, non_blocking=True),
            )
        else:
            x_t, x_n, y_t, y_n = (
                x_t.to(self.config.device),
                x_n.to(self.config.device),
                y_t.to(self.config.device),
                y_n.to(self.config.device),
            )

        return x_t, x_n, y_t, y_n

    def _learning_rate(self, cur_iter: int):
        # learning rate decay scheduler (cosine with warmup)

        # 1) linear warmup for warmup_iters steps
        # 2) if iter > lr_decay_iters, return min learning rate
        # 3) in between, use cosine decay down to min learning rate
        if not self.config.decay_lr:
            return self.config.learning_rate
        if cur_iter < self.config.warmup_iters:
            return self.config.learning_rate * cur_iter / self.config.warmup_iters
        if cur_iter > self.config.lr_decay_iters:
            return self.config.min_lr

        decay_ratio = (cur_iter - self.config.warmup_iters) / (
            self.config.lr_decay_iters - self.config.warmup_iters
        )
        if not 0 <= decay_ratio <= 1:
            raise Exception(f"{decay_ratio} [0 <= decay_ratio <= 1]")

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (
            self.config.learning_rate - self.config.min_lr
        )

    @torch.no_grad()
    def estimate_loss(self, data):
        out = {}
        self.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x_t, x_n, y_t, y_n = self.get_batch(split)
                with self.ctx:
                    _, loss_t, _, loss_n = self(x_t, x_n, y_t, y_n)
                    loss = loss_t + loss_n
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.train()
        return out

    def crop_block_size(self, block_size):
        if block_size > self.config.block_size:
            raise Exception("block size value invalid")

        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @property
    def device_type(self):
        return "cuda" if "cuda" in self.config.device else "cpu"

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @property
    def tokens_per_iter(self):
        return (
            self.config.gradient_accumulation_steps
            * self.config.batch_size
            * self.config.block_size
        )

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised

        return mfu

    def log_eval(self, iter_num, raw_model, data, lr, running_mfu, best_val_loss):
        losses = self.estimate_loss(data)
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        if self.config.mlflow_log:
            virtual_mem = psutil.virtual_memory()
            mlflow.log_metrics(
                {
                    "train/loss": float(losses["train"]),
                    "val/loss": float(losses["val"]),
                    "lr": lr,
                    "mfu_percent": running_mfu * 100,
                    "system/unified_mem_used_gb": virtual_mem.used / 1e9,
                    "system/unified_mem_pct": virtual_mem.percent,
                },
                step=iter_num,
            )
        if losses["val"] < best_val_loss or self.config.always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "model_args": self.config,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": self.config,
                }
                print(f"saving {self.config.model_id} to {self.config.out_dir}")
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.config.out_dir, f"{self.config.model_id}_ep{iter_num}.pt"
                    ),
                )

    def log_no_eval(self, raw_model, loss, local_iter_num, dt, running_mfu, iter_num):
        # scale back
        lossf = loss.item() * self.config.gradient_accumulation_steps

        # let training loop settle
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(
                self.config.batch_size * self.config.gradient_accumulation_steps,
                dt,
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        
        # Calculate throughput
        tokens_per_sec = self.tokens_per_iter / dt if dt > 0 else 0
        
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
        )
        
        if self.config.mlflow_log:
            virtual_mem = psutil.virtual_memory()
            mlflow.log_metrics(
                {
                    "train/loss_step": lossf,
                    "throughput/tokens_per_sec": tokens_per_sec,
                    "throughput/ms_per_iter": dt * 1000,
                    "mfu_percent": running_mfu * 100,
                    "system/unified_mem_used_gb": virtual_mem.used / 1e9,
                    "system/unified_mem_pct": virtual_mem.percent,
                },
                step=iter_num,
            )

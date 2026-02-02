import torch

from nanoGPT.config.train_dota2 import *
from nanoGPT.dota2.data.data_train import get_val_meta
from nanoGPT.model.GPT import GPT, GPTConfig
from nanoGPT.model.run_generator import InferencePars, RunGenerator
from nanoGPT.model.torch_config import set_torch_config

model_id = "ckpt_8_8_256_1024_36384_ep36000.pt"
out_dir = f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/ckpt/dota2_{data_version}"

val, meta = get_val_meta()
print(len(val))

set_torch_config()
gpt_conf = GPTConfig(**model_args)
model = GPT(gpt_conf, meta)
model.load_ckpt(Path(out_dir, model_id))
model = torch.compile(model)  # todo: , mode="reduce-overhead" "max-autotune"

inf_pars = InferencePars(num_samples=1000)
run_generator = RunGenerator(model, inf_pars)
run_generator(val, out=True)

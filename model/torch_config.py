import torch


def set_torch_config():
    torch.manual_seed(999)
    torch.cuda.manual_seed(999)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import torch
from datetime import datetime
from typing import List, Tuple, Dict, Any
import os

Tensor = torch.Tensor
Module = torch.nn.Module
FP32 = torch.float32

T = 500
TRAJ_LEN = 512

### Training Control -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
init_lr = 1e-4
lr_reduce_factor = 0.5
lr_reduce_patience = 50
batch_size = 50
epochs = 1000
log_interval = 10
mov_avg_interval = 15 * T

log_dir = f"./Runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
save_dir = log_dir

### Dataset and Model Configs -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
dataset_args = {
    "max_len": TRAJ_LEN,
    "load_path": "./dataset.pth",
}

ddim_skip_step = 20
diffusion_args = {
    "min_beta": 0.0001,
    "max_beta": 0.05,
    "max_diffusion_step": T,
    "scale_mode": "quadratic",
}
use_ddim = ddim_skip_step != 1
if use_ddim:
    diffusion_args["skip_step"] = ddim_skip_step
actual_diff_step = T // ddim_skip_step + 1

# TW MultiSeq Add
model_args = {
    "in_c": 12,  # input trajectory encoding channels
    "out_c": 2,
    "diffusion_steps": T,  # maximum diffusion steps
    "c_list": [128, 128, 128, 256],  # channel schedule of stages, first element is stem output channels
    "blocks": ["RRRR", "RRRR", "RRRR"],  # number of resblocks in each stage
    "embed_c": 64,  # channels of mix embeddings
    "expend": 4,  # number of heads for attention
    "dropout": 0.0,  # dropout
}

# -*- coding: utf-8 -*-

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
from torch import nn 


#device = torch.device('cpu')
device = torch.device("cuda")
model = Unet(
    dim = 32,
    dim_mults = (1, 2, 2, 4),
    groups = 8 # due to the dim numbers
).to(device) 

model = nn.DataParallel(model)

model.to(device)

op_num = 18
konw_op_num = 0


diffusion = GaussianDiffusion(
    model,
    timesteps = 1000,   # number of diffusion steps
    unmask_number=konw_op_num+1,
    loss_type = 'l2'    # L1 or L2
).to(device)

trainer = Trainer(
    diffusion,
    folder = 'traj_AIB9', 
    system = 'AIB9_REMD_T_full_100000ps_0.2ps',
    train_batch_size = 128,
    train_lr = 1e-5,
    train_num_steps = 2000000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    op_number = op_num,
    fp16 = False                       # turn on mixed precision training with apex
)


#trainer.load(24)
trainer.train()




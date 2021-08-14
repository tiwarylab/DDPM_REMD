# -*- coding: utf-8 -*-

import torch
from torch import nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Dataset_traj, cycle, num_to_groups
import numpy as np
from torch.utils import data

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
).to(device)#.cuda()

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
model_id = 20
trainer.load(model_id)

training_ds = trainer.ds
sample_ds = Dataset_traj('traj_AIB9', 'AIB9_REMD_T_full_100000ps_0.2ps')
num_sample = 20000 
batch_size = 1280
sample_dl = cycle(data.DataLoader(sample_ds, batch_size = batch_size, shuffle=True, pin_memory=True))
batches = num_to_groups(num_sample, batch_size)

all_ops_list = list(map(lambda n: trainer.ema_model.sample(trainer.op_number, batch_size=n, samples = next(sample_dl).cuda()[:n, :]), batches))
all_ops = torch.cat(all_ops_list, dim=0).cpu()
all_ops = trainer.rescale_sample_back(all_ops)
np.save( str(trainer.RESULTS_FOLDER /  f'samples-{model_id}'), all_ops.numpy())

print( str(trainer.RESULTS_FOLDER /  f'samples-{model_id}') )


# -*- coding: utf-8 -*-

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Dataset_traj, cycle, num_to_groups
import torch
from torch import nn 

device = torch.device("cuda")

# define the U-net structure
model = Unet(
    dim = 32,                   
    dim_mults = (1, 2, 2, 4 ),   
    groups = 8 
)
model = nn.DataParallel(model)
model.to(device)

# define diffusion model
op_num = 18  
konw_op_num = 0
diffusion = GaussianDiffusion(
    model,                        # U-net model
    timesteps = 1000,             # number of diffusion steps
    unmask_number=konw_op_num+1,  # the dimension of x2 in P(x1|x2)
    loss_type = 'l2'              # L1 or L2
).to(device)

#set training parameters
trainer = Trainer(
    diffusion,                                   # diffusion model
    folder = 'traj_AIB9',                        # folder of trajectories
    system = 'AIB9_REMD_T_full_100000ps_2.0ps',  # name of the trajectory, the file {system}_traj.npy will be used as the training set
    train_batch_size = 128,                      # training batch size
    train_lr = 1e-5,                             # learning rate
    train_num_steps = 2000000,                   # total training steps
    gradient_accumulate_every = 1,               # gradient accumulation steps
    ema_decay = 0.995,                           # exponential moving average decay
    op_number = op_num,
    fp16 = False                                 # turn on mixed precision training with apex
)

# load trained model
model_id = 30     
trainer.load(model_id) 
# start training
#trainer.train()



# Most of this implementation is based on 
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

import torch
import torch.nn as nn

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def update_linear_schedule(optimizer, sampled_timesteps, total_timesteps, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (sampled_timesteps / float(total_timesteps)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def binary2int(binary, max_action_dim):
    d = 0
    for i, value in enumerate(reversed(binary)):
        if value.item():
            d += 2**i

    if max_action_dim <= d:
        d = max_action_dim-1
    return d
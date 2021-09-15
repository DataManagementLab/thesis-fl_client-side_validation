import torch

def gradient_noise(shape, min=-1., max=1., scale=1/10):
    return ((min - max) * torch.rand(shape) + max) * scale

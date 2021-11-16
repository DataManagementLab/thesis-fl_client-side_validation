import torch

def gradient_noise(shape, min=-1., max=1., scale=1/10, device='cpu'):
    return ((min - max) * torch.rand(shape, device=device) + max) * scale

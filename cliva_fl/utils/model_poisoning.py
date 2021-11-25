import torch

def gradient_noise(shape, min=-1., max=1., scale=1/10, device='cpu'):
    return ((min - max) * torch.rand(shape, device=device) + max) * scale

def rand_item(shape):
    res = ()
    for i in shape:
        res += (torch.randint(i, (1,)).item(),)
    return res

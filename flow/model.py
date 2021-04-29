
import torch.nn as nn
import torch.nn.functional as F

LAYER_PREFIX = 'layer'

class ReLuMLP(nn.Module):
    def __init__(self, layers):
        super(ReLuMLP, self).__init__()
        for i in range(len(layers)-1):
            setattr(self, f'{LAYER_PREFIX}{i+1}', nn.Linear(layers[i], layers[i+1], bias=True))
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        i = 1
        while hasattr(self, f'{LAYER_PREFIX}{i}'):
            x = F.relu(getattr(self, f'{LAYER_PREFIX}{i}')(x))
            i += 1
            
        return x
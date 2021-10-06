
import torch.nn as nn
import torch.nn.functional as F

LAYER_PREFIX = 'layer'

class ReLuMLP(nn.Module):
    def __init__(self, layers):
        super(ReLuMLP, self).__init__()

        self.num_layers = len(layers)-1
        for i in range(self.num_layers):
            setattr(self, f'{LAYER_PREFIX}{i+1}', nn.Linear(layers[i], layers[i+1], bias=True))
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)

        for i in range(1, self.num_layers+1):
            l = getattr(self, f'{LAYER_PREFIX}{i}')
            if False and i == self.num_layers:
                x = F.softmax(l(x), dim=1)
            else:
                x = F.relu(l(x))
            
        return x
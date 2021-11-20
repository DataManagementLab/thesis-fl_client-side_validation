
import torch.nn as nn
import torch.nn.functional as F

class ReLuMLP(nn.Module):
    def __init__(self, layers):
        super(ReLuMLP, self).__init__()

        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
import torch.nn as nn
from collections import defaultdict
from functools import partial

# Inspired by https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af

# temp imports
import sys

def register_activation_hooks(model, module_types=[nn.Linear, nn.Conv2d]):
    activations = defaultdict(list)

    def save_activation(name, module, input, output): 
        activations[name].append(output.detach())

    for name, module in model.named_modules():
        if type(module) in module_types:
            module.register_forward_hook(partial(save_activation, name))
    
    return activations


def register_gradient_hooks(model, module_types=[nn.Linear, nn.Conv2d]):
    gradients = defaultdict(lambda: [None, None, None])

    def save_gradients(name, module, grad_input, grad_output):
        gradients[name][0] = grad_input[0].detach().clone()

    for name, module in model.named_modules():
        if type(module) in module_types:
            module.register_full_backward_hook(partial(save_gradients, name))
    
    return gradients
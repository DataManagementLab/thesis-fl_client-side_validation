#!/usr/bin/env python3
# coding: utf-8

""" TODO
[X] Validate next Weights
[ ] More granular validation (manual linear forward pass)
[ ] Save and validate bias gradients
[ ] Show if we can detect attacks

Performance Parameters:
- Model Size
- Batch Size
"""


# IMPORTS
import sys, time, os, random, gc, logging
from logging import basicConfig, debug, info, warning, error
import numpy as np
from pathlib import Path
import torch, json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from copy import deepcopy
import math
from collections import defaultdict
from functools import partial

basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M', level=logging.DEBUG)

info('IMPORTS DONE')


# torch.manual_seed(0)

# HELPER METHODS
def vc(res): 
    return "\U00002705" if res else "\U0000274C"

def validate_buffer(buffer):
    method = 'baseline'

    for index, (data, target, model_state_dict, next_model_state_dict, optimizer_state_dict, loss, activations, gradients) in buffer.items():
        model = M(layers)
        next_model = M(layers)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        model.load_state_dict(model_state_dict)
        next_model.load_state_dict(next_model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        validate_batch(
            method=method,
            data=data, 
            target=target, 
            activations=activations, 
            gradients=gradients, 
            model=model, 
            optimizer=optimizer, 
            loss=loss, 
            next_model=next_model,
            index=index,
            verbose=False)
        # break

def validate_batch(method='baseline', **args):

    # print('METHOD:', method)
    # print('ARGS:', args.keys())

    if method == 'baseline':
        validate_batch_baseline(**args)
    elif method == 'freivald':
        validate_batch_freivald(**args)

def validate_batch_baseline(data, target, activations, gradients, model, optimizer, loss, next_model, verbose=False, index=None):

    # PREPARE MODEL FOR TRAINING STEP
    model.train()
    hooks_check, activations_check = register_activation_hooks(model)
    
    # TRAIN THE MODEL
    optimizer.zero_grad()
    output = model(data)
    loss_check = loss_fn(output, target)
    loss_check.backward()
    optimizer.step()
    
    size_print = 16
    
    # VALIDATE ACTIVATIONS
    if verbose: print('  ACTIVATIONS:')
    act_total = True
    for key, val in activations_check.items():
        if verbose: print(f'    {key} (n: {val[0].shape[0]})')
        for act_check, act in zip(val, activations[key]):
            act_diff = (torch.mean(torch.abs(act_check - act), dim=1)*100).tolist()
            act_print = [vc(ad == 0.0) for ad in act_diff]
            for ad in act_diff: act_total &= ad == 0.0
            len_print = math.ceil(len(act_print) / size_print)
            if verbose:
                for i in range(len_print):
                    print('      ' + ' '.join(act_print[size_print*(i):size_print*(i+1)]))
    
    # VALIDATE LOSS
    loss_diff = abs(loss_check.item()-loss.item())/abs(loss_check.item())*100
    loss_valid = loss_diff == 0.0
    if verbose: print('  LOSS:\n    {} DIFF[{}, {}] = {}%'.format(vc(loss_valid), loss_check.item(), loss.item(), loss_diff))
    
    # VALIDATE GRADIENT
    if verbose: print('  GRADIENTS:')
    grad_total = True
    gradients_check = {key: getattr(model, key).weight.grad for key in activations.keys()}
    for key, grad_check in gradients_check.items():
        grad_diff = torch.mean(torch.abs(grad_check - gradients[key][0]))*100
        grad_valid = grad_diff == 0.0
        grad_total &= grad_valid.item()
        if verbose: print('    {} {} [{}%]'.format(vc(grad_valid), key, grad_diff))
    
    # VALIDATE WEIGHTS
    if verbose: print('  WEIGHTS:')
    weight_total = True
    for (l, weight), next_weight in zip(model.named_parameters(), next_model.parameters()):
        weight_diff = torch.sum(torch.abs(weight - next_weight))*100
        weight_valid = weight_diff == 0.0
        weight_total &= weight_valid.item()
        if verbose: print('    {} {}'.format(vc(weight_valid), l))

    # if not verbose: 
    #     if index: print(f'batch {index:04d}', end=' ')
    #     print(f'act: {vc(act_total)}; loss: {vc(loss_valid)}; grad: {vc(grad_total)}; weights: {vc(weight_total)}')
    
    if verbose: print()

def freivald(A, B, C, bias=None, rtol=1e-05, atol=1e-08, details=True):
    """
    Freivalds' algorithm to check if AB = C

    Avoid errors because of precision in float32 with '> 1e-5'
    REF: https://discuss.pytorch.org/t/numerical-difference-in-matrix-multiplication-and-summation/28359
    """
    if bias is None: bias = 0.
    r_max = 509090009
    #r = (torch.randint(r_max, (B.shape[1],)) % 2).float() 
    r = torch.ones(B.shape[1])
    ABr = torch.mv(A,torch.mv(B, r))
    Cr = torch.mv(C - bias, r)

    if details:
        diff = torch.isclose(ABr, Cr, rtol=rtol, atol=atol)
        wrong = list()
        for i in range(diff.shape[0]): 
            # print(vc(diff[i]), end='')
            if not diff[i]:
                wrong.append((ABr[i], Cr[i]))
        # print(' - ', end='')
        # if len(wrong) != 0: print()
        for a, b in wrong:
            print(a.item(), b.item(), (a-b).item(), f"{(torch.abs(a-b)/torch.abs(b)*100).item()} %")
    return torch.allclose(ABr, Cr, rtol=rtol, atol=atol)

def baseline(A, B, C, bias=None, rtol=1e-05, atol=1e-08, details=True):
    """
    Normal forward pass
    """
    if bias is None: bias = 0.
    C_ = torch.mm(A,B) + bias
    return torch.allclose(C, C_, rtol=rtol, atol=atol)

def validate_batch_freivald(data, target, activations, gradients, model, optimizer, loss, next_model, verbose=False, index=None):

    optimizer.zero_grad()

    # VALIDATE ACTIVATIONS
    save_input = dict()
    if verbose: print('  ACTIVATIONS:')
    act_total = True
    data = data.view(-1, 28 * 28)
    for key, val in activations.items():
        save_input[key] = torch.clone(data)
        layer = getattr(model, key)
        # act_valid = freivald(data, layer.weight.T, val[0], bias=layer.bias, rtol=1e-05, atol=5e-06)
        act_valid = baseline(data, layer.weight.T, val[0], bias=layer.bias, rtol=1e-05, atol=5e-06)
        if verbose: print(f'    {vc(act_valid)} {key} (n: {val[0].shape[0]})')
        act_total &= act_valid
        data = F.relu(val[0])
    
    # VALIDATE LOSS
    loss_input = torch.clone(data)
    loss_input.requires_grad = True
    loss_check = loss_fn(loss_input, target)
    loss_diff = abs(loss_check.item()-loss.item())/abs(loss_check.item())*100
    loss_valid = loss_diff == 0.0
    if verbose: print('  LOSS:\n    {} DIFF[{}, {}] = {}%'.format(vc(loss_valid), loss_check.item(), loss.item(), loss_diff))
    # data.retain_grad()
    loss_check.backward()
    
    # VALIDATE GRADIENT
    if verbose: print('  GRADIENTS:')
    grad_total = True
    C = loss_input.grad.clone()

    for key, (grad_Weight, grad_bias) in reversed(list(gradients.items())) :

        layer = getattr(model, key)
        W = layer.weight.clone()
        b = layer.bias.clone()

        I = save_input[key]
        A = activations[key][0]
        C_a = (A > 0).float() * C

        # grad_W = torch.mm(C_a.T, I)
        grad_b = torch.sum(C_a, dim=0)
        grad_x = torch.mm(C_a, W)

        # grad_valid = freivald(C_a.T, I, grad_Weight, atol=1e-06)
        grad_valid = baseline(C_a.T, I, grad_Weight, atol=1e-06)
        if verbose: print(f'    {vc(grad_valid)} {key} (n: {grad_Weight.shape[0]})')

        layer.weight.grad = grad_Weight
        layer.bias.grad = grad_bias

        C = grad_x
        grad_total &= grad_valid
        
        # next_layer = getattr(next_model, key)
        # W_new = W + va

    # VALIDATE NEW WEIGHTS
    if verbose: print('  WEIGHTS:')
    optimizer.step()
    weight_total = True

    for key in gradients.keys():
        new_layer = getattr(model, key)
        next_layer = layer = getattr(next_model, key)
        W_valid = torch.allclose(new_layer.weight, next_layer.weight)
        b_valid = torch.allclose(new_layer.bias, next_layer.bias)
        if verbose: print(f'    {key} {vc(W_valid)} (weight) {vc(b_valid)} (bias)')
        weight_total &= W_valid & b_valid

    # PRINT RESULT SUMMARY
    if False and not verbose: 
        if index: pass
        print(f'batch {index:04d}', end=' ')
        print(f'act: {vc(act_total)}; loss: {vc(loss_valid)}; grad: {vc(grad_total)}; weight: {vc(weight_total)}')

    

# LOAD DATASET

# Settings
num_workers = 0
batch_size_train = 64 # 64
batch_size_test = 1000 # 1000

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='datasets', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='datasets', train=False, download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, num_workers=num_workers)

info('DATA LOADED')

# CREATE MODEL

# Settings
layers = [784, 512, 512, 10]

# Model
class M(nn.Module):
    def __init__(self, layers):
        super(M, self).__init__()
        for i in range(len(layers)-1):
            setattr(self, f'l{i+1}', nn.Linear(layers[i], layers[i+1], bias=True))
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        i = 1
        while hasattr(self, f'l{i}'):
            x = F.relu(getattr(self, f'l{i}')(x))
            i += 1
            
        return x

model = M(layers)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

info('MODEL CREATED')

# TRAIN MODEL

def register_activation_hooks(model, selection = dict()):
    activations = {}
    hooks = {}
    def get_hook_fn(l): return lambda model, input, output: activations[l].append(output.detach())

    i = 1
    while hasattr(model, f'l{i}'):
        layer = f'l{i}'
        hooks[layer] = getattr(model, layer).register_forward_hook(
            get_hook_fn(layer)
        )
        activations[layer] = list()
        i += 1
    return (hooks, activations)

def register_gradient_hooks(model, module_types=[nn.Linear]):
    gradients = defaultdict(list)

    def save_gradients(name, module, grad_input, grad_output): 
        print(name, type(module))
        # print(module.grad)
        print('grad_input', grad_input[0].shape, type(grad_input))
        #for x in grad_input: print(x.shape)
        print('grad_output', grad_output[0].shape, type(grad_output))
        #for x in grad_output: print(x.shape)
        for i, inp in enumerate(grad_input): 
           print("Input #", i, inp.shape)
        gradients[name].append([x.shape for x in grad_input])

    for name, module in model.named_modules():
        if type(module) in module_types:
            module.register_full_backward_hook(partial(save_gradients, name))
    
    return gradients

info('START MODEL TRAINING')

model.train()
n_epochs = 1

log_buffer = dict()
log_buffer_max = 100

train_start_time = time.time()

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0

    batch_id = 1

    train_batch_start = time.time()
    
    for index, (data, target) in enumerate(train_loader):
        
        hooks, activations = register_activation_hooks(model)
        #gradients = register_gradient_hooks(model)
        
        # LOGGING
        model_state_dict = deepcopy(model.state_dict())
        optimizer_state_dict = deepcopy(optimizer.state_dict())
        
        # TRAINING
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # break
        gradients = dict()
        for key in activations.keys():
            layer = getattr(model, key)
            gradients[key] = (layer.weight.grad, layer.bias.grad)
        # gradients = {key: (getattr(model, key).weight.grad, getattr(model, key).bias.grad) for key in activations.keys()}

        next_model_state_dict = deepcopy(model.state_dict())

        log_buffer[index] = (data, target, model_state_dict, next_model_state_dict, optimizer_state_dict, loss, deepcopy(activations), deepcopy(gradients))

        activations = {key: list() for key in activations.keys()}
        
        train_loss += loss.item()*data.size(0)

        if len(log_buffer) >= log_buffer_max:
            train_batch_end = time.time()
            
            gc.collect()
            
            validate_batch_start = time.time()
            validate_buffer(log_buffer)
            validate_batch_end = time.time()

            info(f"Time\t{(train_batch_end - train_batch_start):.4f} training\t{(validate_batch_end - validate_batch_start):.4f} validation")

            log_buffer = dict()
            gc.collect()
            batch_id += 1
            train_batch_start = time.time()
            # break

    if len(log_buffer) >= log_buffer_max:
        validate_buffer(log_buffer)
        log_buffer = dict()
    
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

train_end_time = time.time()

print(f'Execution time: {(train_end_time - train_start_time):.4f} sec')


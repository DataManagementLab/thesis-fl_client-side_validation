#!/usr/bin/env python3

""" TODO
[X] Validate next Weights
[ ] More granular validation (manual linear forward pass)
[ ] Save and validate bias gradients
[ ] Show if we can detect attacks

Performance Parameters:
- Model Size
- Batch Size
"""

import torch, sys

import logging as log
log.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%m/%d/%Y %H:%M', 
    level=log.DEBUG)

from flow.dataset import get_dataloader_MNIST
from flow.model import ReLuMLP
from flow.experiment import Experiment
from flow.utils import partial_class

# SETTINGS
num_workers = 0
n_epochs = 5
batch_size_train = 64 # 64
batch_size_test = 1000 # 1000
validation_type = 'extract'
validation_method = 'matmul'
layers = [784, 512, 512, 512, 512, 10]

if __name__ == "__main__":

    # GET DATA
    log.info('Loading Datasets')
    train_loader = get_dataloader_MNIST(batch_size_train, train=True, num_workers=num_workers)
    test_loader = get_dataloader_MNIST(batch_size_test, train=False, num_workers=num_workers)


    # GET MODEL, OPTIMIZER, LOSS
    log.info('Creating Model')
    model_builder = partial_class(ReLuMLP, layers)

    log.info('Creating Optimizer and Loss')
    optimizer_builder = partial_class(torch.optim.SGD, lr=0.01, momentum=0.9)
    loss_fn_builder = torch.nn.CrossEntropyLoss
    
    log.info('Creating Experiment')
    exp = Experiment(model_builder, optimizer_builder, loss_fn_builder, train_loader, validation_type=validation_type, validation_method=validation_method)

    log.info('Setting experiment seed to "0"')
    # exp.seed(0)

    log.info('Running Experiment')
    exp.run(n_epochs, max_buffer_len=50)
    exp.stats()

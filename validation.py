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

# SETTINGS
num_workers = 0
batch_size_train = 64 # 64
batch_size_test = 1000 # 1000

if __name__ == "__main__":

    # GET DATA
    log.info('Loading Datasets')
    train_loader = get_dataloader_MNIST(batch_size_train, train=True)
    test_loader = get_dataloader_MNIST(batch_size_test, train=False)


    # GET MODEL, OPTIMIZER, LOSS
    log.info('Creating Model')
    layers = [784, 512, 512, 10]
    model = ReLuMLP(layers)
    log.info('Creating Optimizer and Loss')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    sys.exit()
    log.info('Creating Experiment')
    exp = Experiment(model, optimizer, loss, dataset, validation_method)
    exp.seed(0)
    log.info('Running Experiment')
    exp.run(n_epochs, max_buffer_len=50)

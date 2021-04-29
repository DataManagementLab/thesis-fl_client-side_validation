# PREVIOUS IMPORTS
# import sys, time, os, random, gc, logging
# from logging import basicConfig, debug, info, warning, error
# import numpy as np
# from pathlib import Path
# import torch, json
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from copy import deepcopy
# import math

# LIB IMPORTS
import torch
from functools import partial

# INIT LOGGER
import logging as log

log.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%m/%d/%Y %H:%M', 
    level=log.DEBUG)

# MODULE IMPORTS
from flow import validation
from flow.utils import ValidationSet, register_activation_hooks, register_gradient_hooks

class Experiment:
    def __init__(self, model, optimizer, loss, dataset, validation_method):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset
        self.set_validation_method(validation_method)

        self.buffer = dict()
        log.debug('Experiment initialized.')
    
    def set_validation_method(self, validation_method):
        assert hasattr(validation, validation_method)
        self.validation_method = validation_method
        self.validation_fn = getattr(validation, validation_method)
        log.debug(f'Set validation method to {validation_method}')

    def seed(self, seed):
        torch.manual_seed(seed)
        log.debug(f'Set seed to {seed}')

    def run(self, n_epochs, max_buffer_len=100):
        self.model.train()
        activations = register_activation_hooks(model)
        gradients = register_gradient_hooks(model)
        
        train_buffer_start = time.time()
        for epoch in range(n_epochs):
            # monitor training loss
            train_loss = 0.0
            batch_time_train = 0.0
            batch_time_valid = 0.0
            
            for index, (data, target) in enumerate(train_loader):
                # SAVE TO SET
                vset = ValidationSet(epoch, batch, self.validation_method)
                vset.set_data(data, target)
                vset.set_model_start(model)
                vset.set_optimizer(optimizer)
                
                # TRAINING
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                for key in gradients.keys():
                    l =  getattr(model, key)
                    gradients[key].append(l.weight.grad)
                    gradients[key].append(l.bias.grad)

                # SAVE TO SET
                vset.set_loss(loss)
                vset.set_activations(activations)
                vset.set_gradients(gradients)
                vset.set_model_end(model)
                buffer[index] = vset

                # EMPTY ACT & GRAD FOR NEXT BATCH
                activations = defaultdict(list)
                gradients = defaultdict(list)
                
                train_loss += loss.item()*data.size(0)

                if len(buffer) >= max_buffer_len:
                    train_buffer_end = time.time()
                    
                    gc.collect() # Free unused memory
                    
                    validate_buffer_start = time.time()
                    self.validate(buffer)
                    validate_buffer_end = time.time()

                    train_buffer_time = train_batch_end - train_batch_start
                    validate_buffer_time = validate_buffer_end - validate_buffer_start
                    log.info(f"Time\t{train_buffer_time:.4f} training\t{validate_buffer_time:.4f} validation")

                    batch_time_train += train_buffer_time
                    batch_time_valid += validate_buffer_time

                    buffer = dict()
                    gc.collect() # Free unused memory
                    train_batch_start = time.time()
                    break
            
            train_loss = train_loss/len(train_loader.dataset)
            log.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

            log.info(f"Epoch Times\t{train_buffer_time:.4f} training\t{validate_buffer_time:.4f} validation")

    def validate(self, buffer):
        for index, (data, target, model_state_dict, next_model_state_dict, optimizer_state_dict, loss, activations, gradients) in buffer.items():
            model = M(layers)
            next_model = M(layers)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            model.load_state_dict(model_state_dict)
            next_model.load_state_dict(next_model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)

            validate_batch(
                method=self.validation_method,
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

    def stats():
        pass

    def reset():
        self.buffer = dict()
        gc.collect()
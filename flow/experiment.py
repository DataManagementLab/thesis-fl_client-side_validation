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

# PROFILER
import cProfile
from numpy import real

# LIB IMPORTS
import torch
import time, gc, sys
from functools import partial
from collections import defaultdict

# INIT LOGGER
import logging as log

log.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%m/%d/%Y %H:%M', 
    level=log.DEBUG)

# MODULE IMPORTS
from flow import validation
import flow.validation.method
from flow.utils import ValidationSet, register_activation_hooks, register_gradient_hooks
class Experiment:
    def __init__(self, model_builder, optimizer_builder, loss_fn_builder, dataset, **validation_args):
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        self.loss_fn_builder = loss_fn_builder

        self.dataset = dataset
        self.set_validation_fn(**validation_args)

        self.reset()
        log.debug('Experiment initialized.')
    
    def set_validation_fn(self, validation_type, validation_method=''):
        SEPARATOR = '.'

        if validation_type == 'retrain':
            self.validation_id = validation_type
            self.validation_fn = validation.validate_retrain
        elif validation_type == 'extract':
            print(validation_method)
            assert hasattr(validation.method, validation_method)
            validation_method_fn = getattr(validation.method, validation_method)

            self.validation_id = SEPARATOR.join((validation_type, validation_method))
            self.validation_fn = partial(validation.validate_extract, validation_method_fn)
        else:
            log.error(f'Validation type {validation_type} is not supported.')
            return

        log.debug(f'Set validation function to {self.validation_id}')

    def reset(self):
        self.init_model_optimizer_loss()

        self.buffer = dict()
        self.set_total_time_training(0.0)
        self.set_total_time_validation(0.0)
        self.set_raw_time_training(0.0)
        self.set_raw_time_validation(0.0)
    
    def init_model_optimizer_loss(self):
        self.model = self.model_builder()
        self.optimizer = self.optimizer_builder(self.model.parameters())
        self.loss_fn = self.loss_fn_builder()

    def seed(self, seed):
        torch.manual_seed(seed)
        log.debug(f'Set seed to {seed}')

    def run(self, n_epochs, max_buffer_len=100):
        self.model.train()
        activations = register_activation_hooks(self.model)
        gradients = register_gradient_hooks(self.model)
        
        for epoch in range(n_epochs):
            # monitor training loss

            raw_time_training = total_time_training = train_loss = 0.0

            prev_raw_time_training = self.get_raw_time_training()
            prev_raw_time_validation = self.get_raw_time_validation()
            prev_total_time_training = self.get_total_time_training()
            prev_total_time_validation = self.get_total_time_validation()
            
            for batch, (data, target) in enumerate(self.dataset):

                start_total_time_training = time.time()

                # SAVE TO SET
                vset = ValidationSet(epoch, batch)
                vset.set_data(data, target)
                vset.set_model_start(self.model)
                vset.set_optimizer(self.optimizer)

                start_raw_time_training = time.time()
                
                # TRAINING
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

                end_raw_time_training = time.time()

                for key in activations.keys():
                    l =  getattr(self.model, key)
                    gradients[key][1] = l.weight.grad
                    gradients[key][2] = l.bias.grad

                # SAVE TO SET
                vset.set_loss(loss)
                vset.set_activations(activations)
                vset.set_gradients(gradients)
                vset.set_model_end(self.model)
                self.buffer[batch] = vset

                # EMPTY ACT & GRAD FOR NEXT BATCH
                for key in list(activations.keys()): del activations[key]
                for key in list(gradients.keys()): del gradients[key]
                
                train_loss += loss.item()*data.size(0)

                end_total_time_training = time.time()

                raw_time_training += end_raw_time_training - start_raw_time_training
                total_time_training += end_total_time_training - start_total_time_training

                if len(self.buffer) >= max_buffer_len or len(self.dataset.dataset) == batch+1:
                    
                    gc.collect() # Free unused memory

                    # print(f"\traw_time_training: {raw_time_training:.4f}")
                    self.add_raw_time_training(raw_time_training)

                    # print(f"\ttotal_time_training: {total_time_training:.4f}")
                    self.add_total_time_training(total_time_training)
                    
                    start_total_time_validation = time.time()
                    #cProfile.runctx('self.validate(self.buffer)', globals(), locals())
                    #.run('self.validate(self.buffer)', 'profile.txt')
                    self.validate(self.buffer)
                    end_total_time_validation = time.time()

                    total_time_validation = end_total_time_validation - start_total_time_validation

                    # print(f"\ttotal_time_validation: {total_time_validation:.4f}")
                    self.add_total_time_validation(total_time_validation)

                    self.buffer = dict()
                    gc.collect() # Free unused memory
                    raw_time_training = total_time_training = 0.0
                    # break
            
            train_loss = train_loss/len(self.dataset.dataset)
            log.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

            log.info(f"Epoch Raw Times\t{self.get_raw_time_training() - prev_raw_time_training:.4f} training\t{self.get_raw_time_validation() - prev_raw_time_validation:.4f} validation")
            log.info(f"Epoch Total Times\t{self.get_total_time_training() - prev_total_time_training:.4f} training\t{self.get_total_time_validation() - prev_total_time_validation:.4f} validation")

    def validate(self, buffer):
        total_time = init_time = validation_time = init_model = init_state_dict = 0.0

        model = self.model_builder()
        next_model = self.model_builder()
        optimizer = self.optimizer_builder(model.parameters())
        loss_fn = self.loss_fn_builder()
        
        for index, vset in buffer.items():
            start_time = time.time()

            model.load_state_dict(vset.get_model_start())
            next_model.load_state_dict(vset.get_model_end())
            optimizer.load_state_dict(vset.get_optimizer())

            mid_time = time.time()

            self.validation_fn(
                model=model, 
                optimizer=optimizer, 
                loss_fn=loss_fn, 
                next_model=next_model,
                index=index,
                verbose=False,
                silent=True,
                **vset.get_dict()
            )
            end_time = time.time()
            total_time += end_time - start_time
            init_time += mid_time - start_time
            validation_time += end_time - mid_time
            # break

        # print(f"\traw_time_validation: {validation_time:.4f}")
        self.add_raw_time_validation(validation_time)

    def set_total_time_training(self, time):
        self.total_time_training = time

    def add_total_time_training(self, time):
        self.total_time_training += time

    def get_total_time_training(self):
        return self.total_time_training
    
    def set_total_time_validation(self, time):
        self.total_time_validation = time
    
    def add_total_time_validation(self, time):
        self.total_time_validation += time
    
    def get_total_time_validation(self):
        return self.total_time_validation

    def set_raw_time_training(self, time):
        self.raw_time_training = time

    def add_raw_time_training(self, time):
        self.raw_time_training += time

    def get_raw_time_training(self):
        return self.raw_time_training
    
    def set_raw_time_validation(self, time):
        self.raw_time_validation = time
    
    def add_raw_time_validation(self, time):
        self.raw_time_validation += time
    
    def get_raw_time_validation(self):
        return self.raw_time_validation
    
    def stats(self):
        log.info(f"Raw Times\t{self.raw_time_training:.4f} training\t{self.raw_time_validation:.4f} validation")
        log.info(f"Total Times\t{self.total_time_training:.4f} training\t{self.total_time_validation:.4f} validation")

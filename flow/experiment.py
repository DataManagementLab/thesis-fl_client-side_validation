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
import torch.multiprocessing as multiprocessing
from numpy import real

# LIB IMPORTS
import torch, numpy
import yaml, math, random
import pandas as pd
import time, gc, sys
from functools import partial
from collections import defaultdict
from pathlib import Path

# INIT LOGGER
import logging as log

log.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%m/%d/%Y %H:%M', 
    level=log.INFO)

# MODULE IMPORTS
from flow import training, validation, datasets, models
from flow.utils import ValidationSet, Logger, TimeTracker, logger, register_activation_hooks, register_gradient_hooks, partial_class, load_config, time_tracker
from flow.multiprocessing import start_validators, stop_validators

class Experiment:

    TRAIN_METRICS = [
        'loss',
        'raw_time_training',
        'raw_time_validation',
        'total_time_training',
        'total_time_validation'
    ]

    def __init__(self, model_builder, optimizer_builder, loss_fn_builder, dataset, config_file, run_validation=True, **validation_args):
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        self.loss_fn_builder = loss_fn_builder
        self.time_tracker = TimeTracker()

        self.dataset = dataset
        self.config_file = config_file
        self.set_training_fn()
        self.run_validation = run_validation
        if run_validation:
            self.set_validation_fn(**validation_args)
        else:
            self.async_validators = 0

        self.reset()
        log.debug('Experiment initialized.')
    
    @classmethod
    def from_config(cls, config_file: Path):

        dataset_cnf, model_cnf, optimizer_cnf, loss_fn_cnf, training_cnf, validation_cnf = load_config(
            config_file, 'dataset', 'model', 'optimizer', 'loss_fn', 'training', 'validation')

        # DATASET
        train_loader = getattr(datasets, 'get_dataloader_{}'.format(dataset_cnf['type']))(**dataset_cnf['params_train'])

        # MODEL, OPTIMIZER, LOSS
        log.info('Creating Model')
        model_builder = partial_class(model_cnf['type'], getattr(models, model_cnf['type']), **model_cnf['params'])

        log.info('Creating Optimizer and Loss')
        optimizer_builder = partial_class(optimizer_cnf['type'], getattr(torch.optim, optimizer_cnf['type']), **optimizer_cnf['params'])
        loss_fn_builder = getattr(torch.nn, loss_fn_cnf['type'])
        
        log.info('Creating Experiment')
        exp = Experiment(
            model_builder, 
            optimizer_builder, 
            loss_fn_builder, 
            train_loader, 
            config_file,
            **validation_cnf)
        
        if 'training_method' in training_cnf:
            log.info('Setting experiment training method to "{}"'.format(training_cnf['training_method']))
            exp.set_training_fn(training_cnf['training_method'], training_cnf['training_params'] if 'training_params' in training_cnf else dict())

        if 'seed' in training_cnf:
            log.info('Setting experiment seed to "{}"'.format(training_cnf['seed']))
            exp.seed(training_cnf['seed'])
        
        return exp
    
    def set_validation_fn(self, validation_type, validation_method='', async_validators=0, async_disk_queue=False, **validation_kwargs):
        SEPARATOR = '.'

        if validation_type == 'retrain':
            self.validation_id = validation_type
            self.validation_fn = validation.validate_retrain
        elif validation_type == 'extract':
            assert hasattr(validation.method, validation_method)
            log.info(f'Using validation method {validation_method}')
            validation_method_fn = getattr(validation.method, validation_method)

            self.validation_id = SEPARATOR.join((validation_type, validation_method))
            self.validation_fn = partial(validation.validate_extract, validation_method_fn, **validation_kwargs)
        else:
            log.error(f'Validation type {validation_type} is not supported.')
            return
        
        self.async_validators = async_validators
        self.async_disk_queue = async_disk_queue

        log.debug(f'Set validation function to {self.validation_id}')
    
    def set_training_fn(self, training_method: str = 'no_attack', training_params: dict = dict()):
        assert hasattr(training, training_method)
        self.training_fn = getattr(training, training_method)
        self.training_params = training_params
        log.debug(f'Set training function to {training_method}')

    def reset(self):
        self.init_model_optimizer_loss()

        self.buffer = dict()
        self.time_tracker.clear()
        self.time_tracker.init_timeframes()
    
    def init_model_optimizer_loss(self):
        self.model = self.model_builder()
        self.optimizer = self.optimizer_builder(self.model.parameters())
        self.loss_fn = self.loss_fn_builder()

    def seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        log.debug(f'Set seed to {seed}')

    def run(self, n_epochs, max_buffer_len=100, shuffle_batches=False, log_dir=None, use_gpu=False):
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        log.info(f'Running in device: {device}')

        self.time_tracker.start_timeframe('training')
        self.model.train()
        self.model = self.model.to(device)
        activations = register_activation_hooks(self.model)
        gradients = register_gradient_hooks(self.model)

        self.logger = Logger(log_dir)
        self.logger.copy_config(self.config_file)

        assert not self.logger.epoch_exists(0)
        self.logger.save_model(epoch=0, model=self.model)
        train_stats = pd.DataFrame(columns=self.TRAIN_METRICS)

        if self.async_validation:
            # Launch Validators, Lock and Queue
            consumers, queue, lock = start_validators(
                self.async_validators,
                logger=self.logger,
                validation_fn=self.validation_fn, 
                # model_builder=self.model_builder,
                # optimizer_builder=self.optimizer_builder,
                loss_fn_builder=self.loss_fn_builder)
                # time_tracker=self.time_tracker)
            self.async_queue = queue

        if shuffle_batches:
            log.debug('Shuffle batch iterator')
            lst = list(iter(self.dataset))
        else:
            lst = self.dataset
        
        for epoch in range(1, n_epochs + 1):

            assert not self.logger.epoch_exists(epoch), f'Can not overwrite existing epoch {epoch}'

            if shuffle_batches: random.shuffle(lst)
            train_loss = 0.
            
            for batch, (data, target) in enumerate(lst):

                self.time_tracker.start('total_time_training')

                # SAVE TO SET
                vset = ValidationSet(epoch, batch)
                vset.set_data(data, target)
                vset.set_model_start(self.model)
                vset.set_optimizer(self.optimizer)
                self.time_tracker.start('raw_time_training')
                loss = self.training_fn(self.model, self.optimizer, self.loss_fn, data, target, epoch, batch, device, self.logger, **self.training_params)
                self.time_tracker.stop('raw_time_training')

                # GET GRADIENTS
                for key in activations.keys():
                    l =  getattr(self.model, key)
                    if len(gradients[key]) == 0: gradients[key].append(None)
                    gradients[key].append(l.weight.grad.detach().clone().cpu())
                    gradients[key].append(l.bias.grad.detach().clone().cpu())

                # SAVE TO SET
                vset.set_loss(loss.detach().clone().cpu())
                vset.set_activations(activations)
                vset.set_gradients(gradients)
                vset.set_model_end(self.model)
                self.buffer[batch] = vset

                # EMPTY ACT & GRAD FOR NEXT BATCH
                for key in list(activations.keys()): del activations[key]
                for key in list(gradients.keys()): del gradients[key]
                
                train_loss += loss.item()*data.size(0)

                self.time_tracker.stop('total_time_training')

                if len(self.buffer) >= max_buffer_len or len(lst) == batch+1:
                    gc.collect() # Free unused memory
                    if self.run_validation:
                        self.validate(self.buffer)
                    self.buffer = dict()
                    gc.collect() # Free unused memory
                    torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    # break
            
            self.logger.save_model(epoch, self.model)
            
            train_loss = train_loss/len(self.dataset.dataset)

            log.info(f'Epoch {epoch}  \t\tTraining Loss: {train_loss:.6f}')
            log.info(f"Epoch Raw Times  \t{self.time_tracker.get('raw_time_training'):.4f} training\t{self.time_tracker.get('raw_time_validation'):.4f} validation")
            log.info(f"Epoch Total Times\t{self.time_tracker.get('total_time_training'):.4f} training\t{self.time_tracker.get('total_time_validation'):.4f} validation")

            train_stats.loc[epoch] = [
                train_loss,
                self.time_tracker.get('raw_time_training'),
                self.time_tracker.get('raw_time_validation'),
                self.time_tracker.get('total_time_training'),
                self.time_tracker.get('total_time_validation')
            ]

            self.logger.save_stats(train_stats)
            self.logger.log_times(epoch, self.time_tracker.total_times_history)
            self.time_tracker.clear()
        
        self.time_tracker.stop_timeframe('training')
        self.logger.save_timeframe('training', 
        **self.time_tracker.get_timeframe('training', format="%Y-%m-%d %H:%M:%S"))

        del self.logger
        if self.async_validation:
            # Join Validators
            stop_validators(consumers, queue)
            del self.async_queue

    def validate(self, buffer):

        if self.async_validation:
            if self.async_disk_queue:
                msg = self.logger.put_queue(buffer)
            else:
                msg = buffer
            self.async_queue.put(msg)
        else:
            validation.validate_buffer(
                buffer, 
                self.validation_fn, 
                self.model_builder,
                self.optimizer_builder,
                self.loss_fn_builder,
                self.time_tracker,
                self.logger)
    
    @property
    def async_validation(self):
        return self.async_validators > 0
    
    def stats(self):
        print(self.time_tracker)

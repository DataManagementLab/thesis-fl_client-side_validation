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
from copy import deepcopy
import torch.multiprocessing as multiprocessing
from numpy import real

# LIB IMPORTS
import torch, numpy
import torch.nn as nn
import torch.multiprocessing as mp
import yaml, math, random
import pandas as pd
import time, gc, sys
from functools import partial
from collections import defaultdict
from pathlib import Path

# INIT LOGGER
import logging as log

from cliva_fl.utils.validation_buffer import ValidationBuffer

log.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%m/%d/%Y %H:%M', 
    level=log.INFO)

# MODULE IMPORTS
from cliva_fl import training, validation, datasets, models
from cliva_fl.utils import ValidationSet, Logger, TimeTracker, logger, register_activation_hooks, register_gradient_hooks, partial_class, load_config, time_tracker, freivalds_rounds
from cliva_fl.multiprocessing import start_validators, stop_validators

class Experiment:

    TRAIN_METRICS = [
        'loss',
        'raw_time_training',
        'raw_time_validation',
        'total_time_training',
        'total_time_validation'
    ]

    def __init__(self, model_builder, optimizer_builder, loss_fn_builder, dataset, config_file, run_validation=True, monitor_memory=False, **validation_args):
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        self.loss_fn_builder = loss_fn_builder
        self.time_tracker = TimeTracker()
        self.reset()

        self.dataset = dataset
        self.config_file = config_file
        self.set_training_fn()
        self.run_validation = run_validation
        self.monitor_memory = monitor_memory
        if run_validation:
            self.set_validation_fn(**validation_args)
        else:
            self.async_validators = 0

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
    
    def set_validation_fn(self, validation_type, validation_method='', async_validators=0, use_queue=False, async_disk_queue=False, guarantee=None, validation_delay=0, **validation_kwargs):
        SEPARATOR = '.'

        if validation_type == 'retrain':
            self.validation_id = validation_type
            self.validation_fn = partial(validation.validate_retrain, **validation_kwargs)
        elif validation_type == 'extract':
            assert hasattr(validation.method, validation_method)
            log.info(f'Using validation method {validation_method}')
            validation_method_fn = getattr(validation.method, validation_method)
            
            extra_kwargs = dict()
            if guarantee:
                if validation_method == 'freivald':
                    extra_kwargs['n_check'] = freivalds_rounds(len(self.model.layers)/2, guarantee)
            log.info(f"extra_kwargs: {extra_kwargs}")

            self.validation_id = SEPARATOR.join((validation_type, validation_method))
            self.validation_fn = partial(validation.validate_extract, validation_method_fn, **extra_kwargs, **validation_kwargs)
        else:
            log.error(f'Validation type {validation_type} is not supported.')
            return
        
        self.async_validators = async_validators
        self.async_disk_queue = async_disk_queue
        self.validation_delay = validation_delay
        self.use_queue = use_queue or self.async_validation

        log.debug(f'Set validation function to {self.validation_id}')
    
    def set_training_fn(self, training_method: str = 'no_attack', training_params: dict = dict()):
        assert hasattr(training, training_method)
        self.training_fn = getattr(training, training_method)
        self.training_params = training_params
        log.debug(f'Set training function to {training_method}')

    def reset(self):
        self.init_model_optimizer_loss()

        self.time_tracker.clear()
        self.time_tracker.init_timeframes()
    
    def init_model_optimizer_loss(self):
        self.model = self.model_builder()
        self.optimizer = self.optimizer_builder(self.model.parameters())
        self.loss_fn = self.loss_fn_builder()

        self.val_model = self.model_builder()
        self.val_optimizer = self.optimizer_builder(self.val_model.parameters())
        self.val_loss_fn = self.loss_fn_builder()

    def seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        log.debug(f'Set seed to {seed}')

    def run(self, n_epochs, max_buffer_len=100, shuffle_batches=False, log_dir=None, use_gpu=False):
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        log.info(f'Running on device: {device}')

        self.time_tracker.start_timeframe('training')

        # PREPARE MODEL FOR TRAINING
        self.model.to(device)
        self.model.train()
        activations = register_activation_hooks(self.model)
        gradients = register_gradient_hooks(self.model)

        # INITIALIZE LOG DIRECTORY
        self.logger = Logger(log_dir)
        assert not self.logger.epoch_exists(0)
        self.logger.copy_config(self.config_file)
        self.logger.save_model(epoch=0, model=self.model)
        train_stats = pd.DataFrame(columns=self.TRAIN_METRICS)

        # LAUNCH VALIDATORS, LOCK AND QUEUE
        if self.use_queue:
            self.queue = mp.Queue()
        if self.async_validation:
            consumers, lock = start_validators(
                self.async_validators,
                self.queue,
                logger=self.logger,
                validation_fn=self.validation_fn,
                loss_fn=self.val_loss_fn,
                validation_delay=self.validation_delay,
                monitor_memory=self.monitor_memory)

        # SHUFFLE BATCHES IF REQUIRED
        if shuffle_batches:
            log.debug('Shuffle batches')
            lst = list(iter(self.dataset))
        else:
            lst = self.dataset
        
        for epoch in range(1, n_epochs + 1):

            assert not self.logger.epoch_exists(epoch), f'Can not overwrite existing epoch {epoch}'

            train_loss = 0.
            if shuffle_batches: random.shuffle(lst)
            buffer = ValidationBuffer(epoch, max_buffer_len)
            buffer.set_init_model_state(self.model)
            self.time_tracker.start('mp_fill_buffer')
            
            for batch, (data, target) in enumerate(lst):

                self.time_tracker.start('total_time_training')
                data = data.view(-1, 28 * 28)

                # SAVE TO SET
                vset = ValidationSet(epoch, batch)
                vset.set_data(data, target)
                vset.set_optimizer_state(self.optimizer)
                self.time_tracker.start('raw_time_training')
                loss = self.training_fn(self.model, self.optimizer, self.loss_fn, data, target, epoch, batch, device, self.logger, **self.training_params)
                self.time_tracker.stop('raw_time_training')

                # GET GRADIENTS
                self.time_tracker.start('raw_time_extract_gradients')
                name = list(activations.keys())[0]
                gradients[name].append(None)
                gradients[name].append(self.model.layers[0].weight.grad.detach().cpu())
                gradients[name].append(self.model.layers[0].bias.grad.detach().cpu())

                # module_types = [nn.Linear, nn.Conv2d]
                # for name, module in self.model.named_modules():
                #     if type(module) in module_types:
                #         if not gradients[name]: 
                #             gradients[name].append(None)
                #             gradients[name].append(module.weight.grad.detach().cpu())
                #             gradients[name].append(module.bias.grad.detach().cpu())
                #             break
                self.time_tracker.stop('raw_time_extract_gradients')

                # for key in activations.keys():
                #     l =  getattr(self.model, key)
                #     if len(gradients[key]) == 0: gradients[key].append(None)
                #     gradients[key].append(l.weight.grad.detach().clone().cpu())
                #     gradients[key].append(l.bias.grad.detach().clone().cpu())

                # SAVE TO BUFFER
                vset.set_loss(loss)
                vset.set_gradients(gradients)
                vset.set_activations(activations)
                vset.set_model_state(self.model)
                buffer.add(batch, vset)

                # EMPTY ACTIVATIONS & GRADIENTS FOR NEXT BATCH
                activations.clear()
                gradients.clear()
                
                train_loss += loss.item()*data.size(0)
                self.time_tracker.stop('total_time_training')

                if buffer.full() or len(lst) == batch+1:
                    self.time_tracker.stop('mp_fill_buffer')
                    gc.collect() # Free unused memory
                    if self.run_validation:
                        self.time_tracker.start('mp_call_validation')
                        self.validate(buffer)
                        self.time_tracker.stop('mp_call_validation')
                    buffer = ValidationBuffer(epoch, max_buffer_len)
                    buffer.set_init_model_state(self.model)
                    gc.collect() # Free unused memory
                    self.time_tracker.start('mp_fill_buffer')
            
            self.time_tracker.stop('mp_fill_buffer')
            
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
            stop_validators(consumers, self.queue)
        if self.use_queue:
            self.queue.close()
            del self.queue

    def validate(self, buffer):

        if self.use_queue:
            self.time_tracker.start('mp_put_queue')
            if self.async_disk_queue:
                msg = self.logger.put_queue(buffer)
            else:
                msg = buffer
            self.queue.put_nowait(msg)
            self.time_tracker.stop('mp_put_queue')
        if not self.async_validation:
            self.time_tracker.start('svr_get_buffer')
            if self.use_queue:
                bffr = self.queue.get()
            else:
                bffr = buffer
            self.time_tracker.stop('svr_get_buffer')
            validation.validate_buffer(
                bffr, 
                self.validation_fn,
                self.val_model,
                self.val_optimizer,
                self.val_loss_fn,
                self.time_tracker,
                self.logger)
    
    @property
    def async_validation(self):
        return self.async_validators > 0
    
    def stats(self):
        print(self.time_tracker)

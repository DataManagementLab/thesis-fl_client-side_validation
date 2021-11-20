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

import yaml, argparse
import logging as log
from pathlib import Path

from cliva_fl.experiment import Experiment

# ARGUMENT PARSING

parser = argparse.ArgumentParser(description='Argument parser for log processing and plot creation')
parser.add_argument('-r', '--repeat', type=int, default=1, required=False, help='Number of times to repeat the experiement.')

args = parser.parse_args()

# LOGGING
log.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%m/%d/%Y %H:%M', 
    level=log.DEBUG)

# SETTINGS
CONFIG_FILE = Path('experiment.yml')

with open(CONFIG_FILE, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

dataset_cnf = config['dataset']
model_cnf = config['model']
optimizer_cnf = config['optimizer']
loss_fn_cnf = config['loss_fn']
training_cnf = config['training']
validation_cnf = config['validation']

log.debug(f'Dataset Config: {dataset_cnf}')
log.debug(f'Model Config: {model_cnf}')
log.debug(f'Optimizer Config: {optimizer_cnf}')
log.debug(f'Loss Function Config: {loss_fn_cnf}')
log.debug(f'Training Config: {training_cnf}')
log.debug(f'Validation Config: {validation_cnf}')

if __name__ == "__main__":
    exp = Experiment.from_config(CONFIG_FILE)
    for i in range(args.repeat):
        log.info(f'Starting experiment {i+1} of {args.repeat}')
        exp.run(training_cnf['n_epochs'], max_buffer_len=training_cnf['max_buffer_len'], shuffle_batches=training_cnf['shuffle_batches'], log_dir=training_cnf['log_dir'], use_gpu=training_cnf['use_gpu'])
        # exp.run(**training_cnf)
        exp.stats()
        exp.reset()

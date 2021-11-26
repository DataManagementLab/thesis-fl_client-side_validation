#!/bin/bash

# TIME PLOTS
python plots/script/line_times.py -c plots/conf/exp_time_method_scalability.yml
python plots/script/line_times.py -c plots/conf/exp_time_guarantee_scalability.yml

# MEMORY PLOTS
python plots/script/line_memory.py -c plots/conf/exp_mem_model_size.yml

# ATTACK PLOTS
python plots/script/line_quality.py -c plots/conf/exp_attack_varying_noise.yml

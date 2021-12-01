#!/bin/bash

# CONCEPT PLOTS
python plots/script/freivald_scalability.py

# TIME PLOTS
python plots/script/line_times.py -c plots/conf/exp_time_method_scalability.yml
python plots/script/line_times.py -c plots/conf/exp_time_guarantee_scalability.yml
python plots/script/timeframes.py -c plots/conf/exp_time_device_concurrency.yml
python plots/script/line_times.py -c plots/conf/exp_time_guarantee_speed.yml
python plots/script/times.py -t -c plots/conf/exp_time_gvfa_train_validate.yml
python plots/script/times.py -t -c plots/conf/exp_time_validation_details.yml

# MEMORY PLOTS
python plots/script/line_memory.py -c plots/conf/exp_mem_model_size.yml

# ATTACK PLOTS
python plots/script/line_quality.py -c plots/conf/exp_attack_varying_noise.yml
python plots/script/line_quality.py -c plots/conf/exp_attack_guarantee_detection.yml

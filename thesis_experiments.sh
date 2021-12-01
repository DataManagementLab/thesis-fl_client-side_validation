#!/bin/bash

N=5

# TIME Experiments
EXP_TIME_METHOD_SCALABILITY=true
EXP_TIME_GUARANTEE_SCALABILITY=true
EXP_TIME_DEVICE_CONCURRENCY=true

# SPACE Experiments
EXP_MEM_MODEL_SIZE=true

# QUALITY Experiments
EXP_ATTACK_VARYING_NOISE=true
EXP_TIME_ATTACK_VARYING_GUARANTEE=true


if $EXP_TIME_METHOD_SCALABILITY; then
echo "===== EXP_TIME_METHOD_SCALABILITY ====="
dirs=(
experiments/time/freivald/l2_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/freivald/l4_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/freivald/l6_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/freivald/l8_512_sync_cpu_bt64_bf32_n1.yml

experiments/time/gvfa/l2_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/gvfa/l4_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/gvfa/l6_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/gvfa/l8_512_sync_cpu_bt64_bf32_n1.yml

experiments/time/matmul/l2_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/matmul/l4_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/matmul/l6_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/matmul/l8_512_sync_cpu_bt64_bf32_n1.yml

experiments/time/retrain/l2_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/retrain/l4_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/retrain/l6_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/retrain/l8_512_sync_cpu_bt64_bf32_n1.yml

experiments/time/submul/l2_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/submul/l4_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/submul/l6_512_sync_cpu_bt64_bf32_n1.yml
experiments/time/submul/l8_512_sync_cpu_bt64_bf32_n1.yml
)
for dir in "${dirs[@]}"; do python run_exp.py -r $N -c "$dir"; done
fi

if $EXP_TIME_GUARANTEE_SCALABILITY; then
echo "===== EXP_TIME_GUARANTEE_SCALABILITY ====="
dirs=(
experiments/time/freivald/l2_512_sync_cpu_bt64_bf32_q99.yml
experiments/time/freivald/l4_512_sync_cpu_bt64_bf32_q99.yml
experiments/time/freivald/l6_512_sync_cpu_bt64_bf32_q99.yml
experiments/time/freivald/l8_512_sync_cpu_bt64_bf32_q99.yml
)
for dir in "${dirs[@]}"; do python run_exp.py -r $N -c "$dir"; done
fi

if $EXP_TIME_DEVICE_CONCURRENCY; then
echo "===== EXP_TIME_DEVICE_CONCURRENCY ====="
dirs=(
experiments/time/gvfa/l2_512_sync_gpu_bt64_bf32_n1_queue.yml
experiments/time/gvfa/l2_512_sync_cpu_bt64_bf32_n1_queue.yml
experiments/time/gvfa/l2_512_async_gpu_bt64_bf32_n1.yml
experiments/time/gvfa/l2_512_async_cpu_bt64_bf32_n1.yml
experiments/time/gvfa/l2_512_sync_gpu_bt64_bf32_n1_noval.yml
experiments/time/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noval.yml
)
for dir in "${dirs[@]}"; do python run_exp.py -r $N -c "$dir"; done
fi

if $EXP_MEM_MODEL_SIZE; then
echo "===== EXP_MEM_MODEL_SIZE ====="
dirs=(
experiments/memory/freivald/l2_512_async_cpu_bt64_bf8_n1.yml
experiments/memory/freivald/l4_512_async_cpu_bt64_bf8_n1.yml
experiments/memory/freivald/l6_512_async_cpu_bt64_bf8_n1.yml
experiments/memory/freivald/l8_512_async_cpu_bt64_bf8_n1.yml

experiments/memory/gvfa/l2_512_async_cpu_bt64_bf8_n1.yml
experiments/memory/gvfa/l4_512_async_cpu_bt64_bf8_n1.yml
experiments/memory/gvfa/l6_512_async_cpu_bt64_bf8_n1.yml
experiments/memory/gvfa/l8_512_async_cpu_bt64_bf8_n1.yml

experiments/memory/matmul/l2_512_async_cpu_bt64_bf8.yml
experiments/memory/matmul/l4_512_async_cpu_bt64_bf8.yml
experiments/memory/matmul/l6_512_async_cpu_bt64_bf8.yml
experiments/memory/matmul/l8_512_async_cpu_bt64_bf8.yml

experiments/memory/retrain/l2_512_async_cpu_bt64_bf8.yml
experiments/memory/retrain/l4_512_async_cpu_bt64_bf8.yml
experiments/memory/retrain/l6_512_async_cpu_bt64_bf8.yml
experiments/memory/retrain/l8_512_async_cpu_bt64_bf8.yml

experiments/memory/submul/l2_512_async_cpu_bt64_bf8.yml
experiments/memory/submul/l4_512_async_cpu_bt64_bf8.yml
experiments/memory/submul/l6_512_async_cpu_bt64_bf8.yml
experiments/memory/submul/l8_512_async_cpu_bt64_bf8.yml
)
for dir in "${dirs[@]}"; do python run_exp.py -r $N -c "$dir"; done
fi

if $EXP_ATTACK_VARYING_NOISE; then
echo "===== EXP_ATTACK_VARYING_NOISE ====="
dirs=(
experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-1.yml
experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-2.yml
experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-3.yml
experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-4.yml
experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-5.yml

experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-1.yml
experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-2.yml
experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-3.yml
experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-4.yml
experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-5.yml

experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-1.yml
experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-2.yml
experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-3.yml
experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-4.yml
experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-5.yml
)
for dir in "${dirs[@]}"; do python run_exp.py -r $N -c "$dir"; done
fi

if $EXP_TIME_ATTACK_VARYING_GUARANTEE; then
echo "===== EXP_TIME_ATTACK_VARYING_GUARANTEE ====="
dirs=(
experiments/time_attack/freivald/l2_512_sync_cpu_bt64_bf32_q25_noise5e-4.yml
experiments/time_attack/freivald/l2_512_sync_cpu_bt64_bf32_q50_noise5e-4.yml
experiments/time_attack/freivald/l2_512_sync_cpu_bt64_bf32_q75_noise5e-4.yml
experiments/time_attack/freivald/l2_512_sync_cpu_bt64_bf32_q90_noise5e-4.yml
experiments/time_attack/freivald/l2_512_sync_cpu_bt64_bf32_q99_noise5e-4.yml

experiments/time_attack/submul/l2_512_sync_cpu_bt64_bf32_q25_noise5e-4.yml
experiments/time_attack/submul/l2_512_sync_cpu_bt64_bf32_q50_noise5e-4.yml
experiments/time_attack/submul/l2_512_sync_cpu_bt64_bf32_q75_noise5e-4.yml
experiments/time_attack/submul/l2_512_sync_cpu_bt64_bf32_q90_noise5e-4.yml
experiments/time_attack/submul/l2_512_sync_cpu_bt64_bf32_q99_noise5e-4.yml
)
for dir in "${dirs[@]}"; do python run_exp.py -r $N -c "$dir"; done
fi

#!/bin/bash

N=5

# TIME Experiments
EXPERIMENT_METHOD_SCALABILITY=true
EXPERIMENT_GUARANTEE_SCALABILITY=false

# SPACE Experiments
EXPERIMENT_BUFFER_SIZE=false

# QUALITY Experiments
EXPERIMENT_VARYING_NOISE=true


if $EXPERIMENT_METHOD_SCALABILITY; then
echo "===== TIME EXPERIMENT_METHOD_SCALABILITY ====="
# echo "===== TIME EXPERIMENT_METHOD_SCALABILITY | Freivald ====="
# python run_exp.py -r $N -c experiments/time/freivald/l2_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/freivald/l4_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/freivald/l6_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/freivald/l8_512_sync_cpu_bt64_bf32_n1.yml

# echo "===== TIME EXPERIMENT_METHOD_SCALABILITY | MatMul ====="
# python run_exp.py -r $N -c experiments/time/matmul/l2_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/matmul/l4_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/matmul/l6_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/matmul/l8_512_sync_cpu_bt64_bf32_n1.yml

echo "===== TIME EXPERIMENT_METHOD_SCALABILITY | Retrain ====="
python run_exp.py -r $N -c experiments/time/retrain/l2_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/retrain/l4_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/retrain/l6_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/retrain/l8_512_sync_cpu_bt64_bf32_n1.yml

# echo "===== TIME EXPERIMENT_METHOD_SCALABILITY | SubMul ====="
# python run_exp.py -r $N -c experiments/time/submul/l2_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/submul/l4_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/submul/l6_512_sync_cpu_bt64_bf32_n1.yml
# python run_exp.py -r $N -c experiments/time/submul/l8_512_sync_cpu_bt64_bf32_n1.yml
fi

if $EXPERIMENT_GUARANTEE_SCALABILITY; then
echo "===== TIME EXPERIMENT_GUARANTEE_SCALABILITY ====="
echo "===== TIME EXPERIMENT_GUARANTEE_SCALABILITY | Freivald ====="
python run_exp.py -r $N -c experiments/time/freivald/l2_512_sync_cpu_bt64_bf32_q99.yml
python run_exp.py -r $N -c experiments/time/freivald/l4_512_sync_cpu_bt64_bf32_q99.yml
python run_exp.py -r $N -c experiments/time/freivald/l6_512_sync_cpu_bt64_bf32_q99.yml
python run_exp.py -r $N -c experiments/time/freivald/l8_512_sync_cpu_bt64_bf32_q99.yml
fi

if $EXPERIMENT_BUFFER_SIZE; then
echo "===== MEMORY EXPERIMENT_BUFFER_SIZE ====="
echo "===== MEMORY EXPERIMENT_BUFFER_SIZE | Freivald ====="
python run_exp.py -r $N -c experiments/memory/freivald/l2_512_async_cpu_bt64_bf8_n1.yml
python run_exp.py -r $N -c experiments/memory/freivald/l2_512_async_cpu_bt64_bf16_n1.yml
python run_exp.py -r $N -c experiments/memory/freivald/l2_512_async_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/memory/freivald/l2_512_async_cpu_bt64_bf64_n1.yml

echo "===== MEMORY EXPERIMENT_BUFFER_SIZE | MatMul ====="
python run_exp.py -r $N -c experiments/memory/matmul/l2_512_async_cpu_bt64_bf8_n1.yml
python run_exp.py -r $N -c experiments/memory/matmul/l2_512_async_cpu_bt64_bf16_n1.yml
python run_exp.py -r $N -c experiments/memory/matmul/l2_512_async_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/memory/matmul/l2_512_async_cpu_bt64_bf64_n1.yml

echo "===== MEMORY EXPERIMENT_BUFFER_SIZE | SubMul ====="
python run_exp.py -r $N -c experiments/memory/submul/l2_512_async_cpu_bt64_bf8_n1.yml
python run_exp.py -r $N -c experiments/memory/submul/l2_512_async_cpu_bt64_bf16_n1.yml
python run_exp.py -r $N -c experiments/memory/submul/l2_512_async_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/memory/submul/l2_512_async_cpu_bt64_bf64_n1.yml
fi

if $EXPERIMENT_VARYING_NOISE; then
echo "===== QUALITY EXPERIMENT_VARYING_NOISE ====="
echo "===== QUALITY EXPERIMENT_VARYING_NOISE | Freivald ====="
python run_exp.py -r $N -c experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-2.yml
python run_exp.py -r $N -c experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-1.yml
python run_exp.py -r $N -c experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-3.yml
python run_exp.py -r $N -c experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-4.yml
python run_exp.py -r $N -c experiments/attack/freivald/l2_512_sync_cpu_bt64_bf32_n1_noise1e-5.yml

# echo "===== QUALITY EXPERIMENT_VARYING_NOISE | GVFA ====="
# python run_exp.py -r $N -c experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-1.yml
# python run_exp.py -r $N -c experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-2.yml
# python run_exp.py -r $N -c experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-3.yml
# python run_exp.py -r $N -c experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-4.yml
# python run_exp.py -r $N -c experiments/attack/gvfa/l2_512_sync_cpu_bt64_bf32_n1_noise1e-5.yml

echo "===== QUALITY EXPERIMENT_VARYING_NOISE | SubMul ====="
python run_exp.py -r $N -c experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-1.yml
python run_exp.py -r $N -c experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-2.yml
python run_exp.py -r $N -c experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-3.yml
python run_exp.py -r $N -c experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-4.yml
python run_exp.py -r $N -c experiments/attack/submul/l2_512_sync_cpu_bt64_bf32_s50_noise1e-5.yml
fi

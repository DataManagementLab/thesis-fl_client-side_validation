#!/bin/bash

N=5

echo "===== TIME EXPERIMENTS ====="
echo "===== TIME EXPERIMENTS | Freivald ====="
python run_exp.py -r $N -c experiments/time/freivald/l2_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/freivald/l4_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/freivald/l6_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/freivald/l8_512_sync_cpu_bt64_bf32_n1.yml

echo "===== TIME EXPERIMENTS | MatMul ====="
python run_exp.py -r $N -c experiments/time/matmul/l2_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/matmul/l4_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/matmul/l6_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/matmul/l8_512_sync_cpu_bt64_bf32_n1.yml

echo "===== TIME EXPERIMENTS | Retrain ====="
python run_exp.py -r $N -c experiments/time/retrain/l2_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/retrain/l4_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/retrain/l6_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/retrain/l8_512_sync_cpu_bt64_bf32_n1.yml

echo "===== TIME EXPERIMENTS | SubMul ====="
python run_exp.py -r $N -c experiments/time/submul/l2_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/submul/l4_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/submul/l6_512_sync_cpu_bt64_bf32_n1.yml
python run_exp.py -r $N -c experiments/time/submul/l8_512_sync_cpu_bt64_bf32_n1.yml

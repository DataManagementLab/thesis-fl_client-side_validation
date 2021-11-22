# Master Thesis von Benedikt Völker

# Install Module

You can install the clivaFL module with pip

    pip install -e .

## Setup

Create a virtual python environment

    pip install virtualenv
    virtualenv venv --python=python3

Enter the virtual environment and install requirements

    source venv/bin/activate
    pip install -r requirements.txt

Now you are good to go.

## Python Profiling

On Linux system install required libraries

    sudo easy_install SquareMap RunSnakeRun

and then run the program with profiler

    python -m cProfile -o out.profile validation.py

The logs can be visualized with

    runsnake out.profile

Reference:

* [https://pythonspot.com/python-profiling/](https://pythonspot.com/python-profiling/)
* [https://docs.python.org/3/library/profile.html](https://docs.python.org/3/library/profile.html)

# Memory Profiler

Install with pip

    pip install memory_profiler

run script with memory profiler (multiprocess)

    mprof run --multiprocess python run_exp.py

show profiler plot

    mprof plot

# Evaluation Plots

1. Evaluate Approaches (Performance & Memory per layer size)

## Parameters

| Parameter    | Values |
|--------------|--------|
| val method   | freivald, gvfa, submul, matmul |
| model size   | # of hidden layers (2,4,6,8) |
| layer size   | 256, 512, 1024, (2048) |
| concurrency  | sync, async |
| device       | CPU, GPU |
| batch size   | 32, 64, 128, (256) |
| buffer size  | 32, 64, 128, (256) |
| guarantee    | 50%, 75%, 90%, 99% |
| n_check      | 1-15 (depends on guarantee & # layers) |
| attack noise | 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 |


## Time

### Scalability of Validation Methods

Measure Validation time for validation methods for different model size

| Parameter    | Values |
|--------------|--------|
| val method   | freivald, gvfa, submul, matmul |
| model size   | 2, 4, 6, 8 |
| layer size   | 512 |
| concurrency  | sync |
| device       | CPU |
| batch size   | 64 |
| buffer size  | 32 |
| n_check      | 1 |

### Scalability of Validation Methods (Fixed Guarantee - Freivalds & GVFA)

Measure Validation time for validation methods for different model size with fixed guarantee

| Parameter    | Values |
|--------------|--------|
| val method   | freivald, gvfa |
| model size   | 2, 4, 6, 8 |
| layer size   | 512 |
| concurrency  | sync |
| device       | CPU |
| batch size   | 64 |
| buffer size  | 32 |
| guarantee    | 99% |

### Validation Time per Guarantee (Fixed #Layers - Freivalds & SubMul)

Measure Validation time for validation methods for different guarantees

| Parameter    | Values |
|--------------|--------|
| val method   | freivald, submul |
| model size   | 2 |
| layer size   | 512 |
| concurrency  | sync |
| device       | CPU |
| batch size   | 64 |
| buffer size  | 32 |
| guarantee    | 50%, 75%, 90%, 99% |

### Training & Validation Time for Sync/Async and CPU/GPU (Freivalds)

Measure Validation time for validation methods for different guarantees

| Parameter    | Values |
|--------------|--------|
| val method   | freivald |
| model size   | 2 |
| layer size   | 512 |
| concurrency  | sync, async |
| device       | CPU, GPU |
| batch size   | 64 |
| buffer size  | 32 |
| guarantee    | 99% |

## Memory

process size = model size * batch size * layer size * buffer size

### Memory Consumption per Buffer Size

Measure memory consumption for differen buffer sizes (with freivalds)

| Parameter    | Values |
|--------------|--------|
| val method   | freivald |
| model size   | 2 |
| layer size   | 512 |
| concurrency  | sync |
| device       | CPU |
| batch size   | 64 |
| buffer size  | 8, 16, 32, 64 |
| n_check      | 1 |

## Detection Quality

### Detection of varying Noise

Measure Detection Quality for different noise level (fixed guarantee)

| Parameter    | Values |
|--------------|--------|
| val method   | freivald |
| model size   | 2 |
| layer size   | 512 |
| concurrency  | sync |
| device       | CPU |
| batch size   | 64 |
| buffer size  | 32 |
| attack noise | 1e-5, 1e-3, 1e-1 |
| guarantee    | 99% |

### Detection Quality per Guarantee

Measure Detection Quality for different guarantee levels (fixed noise)

| Parameter    | Values |
|--------------|--------|
| val method   | freivald |
| model size   | 2 |
| layer size   | 512 |
| concurrency  | sync |
| device       | CPU |
| batch size   | 64 |
| buffer size  | 32 |
| attack noise | 1e-3 |
| guarantee    | 50%, 75%, 90%, 99% |




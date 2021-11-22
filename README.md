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

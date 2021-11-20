from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.0.1'
__package__ = 'flow'

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

long_description = 'flow is a library that was developed during a master thesis' \
                   ' with the goal of leveraging client-side validation of the' \
                   ' training process to eliminate model poisoning attacks in federated learning.'

setup(
    name=__package__,
    version=__version__,
    description='A library for client-side training validation in federated learning.',
    long_description=long_description,
    url='https://github.com/benvoe',
    author="Benedikt Voelker",
    author_email='benedikt@voelker.tech',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith(__package__)],
    zip_safe=False,
    install_requires=requires_list,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ]
)
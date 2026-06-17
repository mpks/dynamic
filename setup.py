#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='dynamic',
    version='0.0.1',
    license="BSD-3-Clause",
    entry_points={
         'console_scripts': [
             'dynamic-extract-dials = dynamic.extract_dials_experiment:main',
             'myedly = dynamic.simulation.cli:main',
        ]
    },
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
)

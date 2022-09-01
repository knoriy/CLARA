#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='clasp',
    version='0.0.1',
    description='Describe Your Cool Project',
    author='Kari Noriy',
    author_email='knoriy72@gmail.com',
    url='https://github.com/knoriy/CLASP',
    install_requires=['pytorch-lightning==1.7.2'],
    packages=find_packages(),
)


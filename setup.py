#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='clara',
    version='0.0.3',
    description='CLARA is designed for multilingual audio representation through a contrastive learning approach. Our aim is to develop a shared representation for various languages and acoustic scenarios. We leverage a rich multilingual audio-text dataset, augmented for diversity. With CLARA, we focus on building a comprehensive model for speech, targeting emotion detection, sound categorisation, and cross-modal retrieval in both zero-shot and few-shot settings. The results demonstrate its potential for universal speech representation that is adaptable to new languages and tasks, minimising reliance on labelled data and enhancing cross-lingual adaptability.',
    author='Kari Noriy',
    author_email='knoriy72@gmail.com',
    url='https://github.com/knoriy/clara',
    install_requires=[
        'pytorch-lightning==2.1.0', 
        'torch==2.1', 
        'torchdata==0.7',
        'librosa',
        'unidecode',
        'inflect',
        'torchaudio',
        'transformers',
        'boto3',
        ],
    packages=find_packages(),
)
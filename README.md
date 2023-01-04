<div align="center">

# CLASP - Contrastive Language-Speech Pretraining

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/knoriy/CLASP/workflows/CI%20testing/badge.svg?branch=master&event=push)

<!--  
Conference   
-->
</div>

## Description

Note: This project is under active development; therefore, we can not guarantee the code base to be perfect or bug-free. Any contribution is welcomed and greatly appreciated.

CLASP is a multilingual neural network designed to identify audio features from natural language. CLASP follows the work CLIP and SimCLR. CLASP can be applied in many text and audio classification tasks, such as language, emotion, instrument and sound.

## How to run

First, install dependencies

```bash
# clone clasp   
git clone https://github.com/knoriy/CLASP.git
cd CLASP

# create conda env
conda env create -f environments/env.yaml

# or

# docker container: Nvidia Docker is required to use with GPU
docker build --no-cache ./environments/ -t knoriy/clasp
docker run -it --rm --gpus=all -v $(pwd):/workspace --name clasp knoriy/clasp

```

By default the container start a juypter note book, to start container in interactive shell mode use:

```bash
docker run -it --rm --gpus=all -v $(pwd):/workspace --name clasp knoriy/clasp bash
```

## train model

```bash

# run module
python clasp/train.py --max_epochs 1 --accelerator gpu --devices 1
# or for distributed
python clasp/train.py --max_epochs 1 --accelerator gpu --strategy ddp --devices 2
# overfit to single batch 
python clasp/train.py --max_epochs 100 --accelerator gpu --overfit_batches 1 --log_every_n_steps 1 --name CLASP
# Debug
python clasp/train.py --max_epochs 1 --accelerator gpu --devices 1 --name CLASP --fast_dev_run True # or --fast_dev_run n where n is the number of step
# predict
python clasp/train.py --max_epochs 100 --accelerator gpu --overfit_batches 1 --testing_stuff True  --limit_predict_batches 1 --logger False

```

## Tensorboard

```bash
tensorboard dev upload --logdir lightning_logs --verbose 0
```

## Install CLASP via pip

Note: This has not been fully tested. If you find any issue please open an issue, with code to replicate the problem.

This CLASP is setup as a package which means you can now easily import any file into any other file, like so:

```python
from clasp.datamodules import WebdatasetDataModule
from clasp.clasp import CLASP
from pytorch_lightning import Trainer

# model
model = CLASP(...)

# data
dataset = WebdatasetDataModule(...)

# train
trainer = Trainer()
trainer.fit(model, datamodule=dataset)

# test using the best model!
trainer.test(ckpt_path='best', datamodule=dataset)
```

### Citation

```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```

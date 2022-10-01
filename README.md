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

CLASP is a multilingual neural network designed to identify audio features from natural language. CLASP follows the work CLIP by OpenAI. CLASP can be applied in many text and audio classification tasks, such as language, emotion, instrument and sound.

## How to run

First, install dependencies

```bash
# clone clasp   
git clone https://github.com/knoriy/CLASP.git

# install clasp   
cd CLASP
pip install -e .   
pip install -r enrioments/requirements.txt
 ```

 Next, navigate to any file and run it.

 ```bash

# run module
python clasp/train.py --max_epochs 1 --accelerator gpu
# or for distributed
python clasp/train.py --max_epochs 1 --accelerator gpu --strategy ddp --devices 8
```

## Tensorboard

```bash
tensorboard dev upload --logdir lightning_logs
```

## Imports

This clasp is setup as a package which means you can now easily import any file into any other file like so:

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

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```

<div align="center">

# CLARA: Multilingual Contrastive Learning for Audio Representation Acquisition

[![Arxiv](http://img.shields.io/badge/Arxiv-2023-B31B1B.svg)](https://arxiv.org/abs/2310.11830)
![CI testing](https://github.com/knoriy/CLARA/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>

## Overview
**CLARA** is designed for multilingual audio representation through a contrastive learning approach. Our aim is to develop a shared representation for various languages and acoustic scenarios. We leverage a rich multilingual audio-text dataset, augmented for diversity. With CLARA, we focus on building a comprehensive model for speech, targeting emotion detection, sound categorisation, and cross-modal retrieval in both zero-shot and few-shot settings. The results demonstrate its potential for universal speech representation that is adaptable to new languages and tasks, minimising reliance on labelled data and enhancing cross-lingual adaptability.

**Note**: This project is in active development. Contributions are encouraged and welcomed.

## Models
We will provide our models for all to use, ready to download from Huggingface. Additionally, we provide models fine-tuned on specific datasets, ensuring optimised performance for specialized tasks. Below, you'll find an organised listing of our base models and their fine-tuned counterparts, complete with download links for each.

| Size     | Parameters | Model Download                                                              |
|----------|------------|-----------------------------------------------------------------------------|
| small    | # M        | [x]()                                                                       |
| medium   | 109 M      | [x]()     |
| large    | # M        | [x]()                                                                       |

### Finetuned model of varous datasets
| FineTuned | Base Model | Model Download                                                     |
|-----------|------------|--------------------------------------------------------------------|
| AudioSet  | medium     | [x]()                                                              |
| Crema-D   | medium     | [x]()                                                              |
| MSWC      | medium     | [x]()                                                              |

If you've fine-tuned CLARA on your dataset and wish to feature it here, please contact us.

## Installation
Clone the repository:
```bash
# clone CLARA   
git clone https://github.com/knoriy/CLARA.git
cd CLARA
```

### Conda
Create a conda environment:

``` bash
# Create conda env
conda env create -f environments/env.yaml
```

### Docker
Build and run the container (Nvidia Docker required for GPU):
``` bash
docker build --no-cache ./environments/ -t knoriy/clara
docker run -it --rm --gpus=all -v $(pwd):/workspace --name clara knoriy/clara
```
By default the container starts a juypter notebook, to start the container in interactive mode, use:

```bash
docker run -it --rm --gpus=all -v $(pwd):/workspace --name clara knoriy/clara bash
```
### Pip

Note: This has not been fully tested. If you find any issue please open an issue, with code to replicate the problem.

This CLARA is setup as a package which means you can now easily import any file into any other file, like so:

``` bash
pip install git+https://github.com/knoriy/CLARA.git
```

## Train model

CLARA is built upon [pytorch-lightning (PL)](https://lightning.ai/docs/pytorch/stable/). For guidance, please refer to the PL CLI documentation.

For a list of all parameters, you can use the following command:

``` bash
python clara/train.py fit --help
```
To fit and train the model on your own data,
``` bash
python clara/train.py fit \
    --trainer path/to/trainer_config.yml \
    --model path/to/model_config.yml \
    --data path/to/data_config.yml
```

We provide some default config files for training CLARA `--data.root_data_path` should be used to direct to tar sharded dataset, this follows the format of [webdataset](https://webdataset.github.io/webdataset/creating/). We currently support locally stored data and those stored on aws S3.

``` bash
python clara/train.py fit \
    --config ./config/config/base.yaml \
    --trainer ./config/config/trainer/base.yaml \
    --model ./config/config/model/pl_clara_100M.yaml \
    --data ./config/config/data/base.yaml \
    --data.root_data_path path/to/dataset/ \
    --data.num_workers 6 \
    --data.batch_size 6 \
    --data.dataset_list ./config/dataset_list.txt \
    --trainer.logger.name clara_100M_FT_RAV \
```
## Eval
### Supported Tasks and Datasets

This project facilitates various audio classification tasks, namely:

- `Emotion`
- `Gender`
- `Sounds`
- `Speech`

Currently, we extend support to the following datasets for each task:

#### Sounds Classification:
- ESC50
- AudioSet
- US8K
- FSD50K

#### Emotion Classification:
- EMNS
- EmoV-DB
- CREMA-D
- RAVDESS

#### Speech Classification:
- MSWC

Utilise these datasets to perform nuanced audio classification across various domains, enhancing your model's understanding and predictive capabilities.

### Zeroshot
``` bash
python clara/eval/test_zeroshot.py \
--model_path path/to/checkpoint.ckpt \
--task emotion \
--dataset_name ravdess \
--root_cfg_path ./config/
```
### Retrieval
``` bash
python clara/eval/test_retrieval.py \
--model_path path/to/checkpoint.ckpt \
--task sounds \
--dataset_name audioset \
--root_cfg_path ./config/
```

## Citation

```bibtex
@article{noriy_clara:_2023,
  title = {{CLARA}: {Multilingual} {Contrastive} {Learning} for {Audio} {Representation} {Acquisition}},
  shorttitle = {{CLARA}},
  author = {Noriy, Kari A. and Yang, Xiaosong and Budka, Marcin and Zhang, Jian Jun},
  note = {arXiv:2310.11830 [cs, eess]},
  url = {http://arxiv.org/abs/2310.11830},
  doi = {10.48550/arXiv.2310.11830},
  year = {2023}
}
```

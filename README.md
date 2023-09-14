<div align="center">

# CLARA: Multilingual Contrastive Learning for Audio Representation Acquisition

[![Paper](http://img.shields.io/badge/Journal-2023-B31B1B.svg)](https://www.nature.com/articles/nature14539)
![CI testing](https://github.com/knoriy/CLASP/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>

## Description
**CLARA** is a model for multilingual speech and audio representation learning using a contrastive approach. The goal is to learn shared representations that generalise across different languages and acoustic conditions. We train our model on a large corpus of diverse multilingual audio data paired with text descriptions. Data augmentation techniques are used to expand the dataset. 

With CLARA, we're striving to shape a core model that encapsulates the intricacies of human speech. This is specifically catered to applications such as emotion detection, sound categorisation, and audio and text retrival tasks in both zero-shot and few-shot environments.

Our result have shown promissing result for acquiring a universal speech representations that transfer well to new languages and tasks. Key benefits include reducing dependence on labelled data and improving cross-lingual generalisation.

**Note**: This project is under active development; therefore, we can not guarantee the code base to be perfect or bug-free. Any contribution is welcomed and greatly appreciated.

## Insallation
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
By default the container start a juypter note book, to start container in interactive shell mode use:

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

For a list of all parametes, pleaseyou can use the following command:

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

We provide some default config files for training CLARA `--data.root_data_path` should be used to direct to tar sharded dataset, this follow the format of [webdataset](https://webdataset.github.io/webdataset/creating/). We currently support localy stored data and those stored on aws S3.

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
### Zeroshot
``` bash

```
### Retrival
``` bash

```

## Citation

```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```

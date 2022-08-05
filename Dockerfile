FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

LABEL MAINTAINER="Kari Noriy knoriy@bournemouth.ac.uk"

RUN apt-get update

## START - Creating user
RUN apt-get -y install sudo
RUN adduser --disabled-password --gecos '' user
RUN adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
## END - Creating user

# RUN apt-get install -y  git \
#                         tree \
# 			nano

## START - Jupyter notebook setup
RUN pip install jupyterlab
## END - Jupyter notebook setup

## START - Python packages
ADD requirements.txt .
RUN pip install -r requirements.txt
## END - Python packages

WORKDIR /workspace
RUN chown user:user /workspace
USER user

ENV type lab

CMD ["bash","-c", "jupyter ${type} --ip 0.0.0.0 --no-browser"]

#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --job-name=laion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --account laion
#SBATCH --output=logs/outs/%x_%j.out
#SBATCH --signal=SIGUSR1@90
#SBATCH --signal=SIGTERM@90

module load openmpi
LOGGER_NAME=CLASP_100M_sounds

MODEL_CONF_PATH=./config/config/model/pl_clasp_100M.yaml
BASE_CONF_PATH=./config/config/base.yaml
TRAINER_BASE_CONF_PATH=./config/config/trainer/base.yaml
TRAINER_CONF_PATH=./config/config/trainer/slurm.yaml
DATA_CONF_PATH=./config/config/data/base.yaml

srun /admin/home-knoriy/miniconda3/envs/hf/bin/python ./scripts/move_to_server.py
#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --job-name=clsp100M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --account laion
#SBATCH --output=logs/outs/%x_%j.out
#SBATCH --signal=SIGUSR1@90
#SBATCH --signal=SIGTERM@90

module load openmpi
srun hostname # for debugging, logs the hostname to the output file

LOGGER_NAME=CLASP_100M

MODEL_CONF_PATH=./config/config/model/pl_clasp_100M.yaml
BASE_CONF_PATH=./config/config/base.yaml
TRAINER_BASE_CONF_PATH=./config/config/trainer/base.yaml
TRAINER_CONF_PATH=./config/config/trainer/slurm.yaml
DATA_CONF_PATH=./config/config/data/base.yaml

srun /admin/home-knoriy/miniconda3/envs/clasp/bin/python /fsx/knoriy/CLASP/scripts/download_from_s3.py --data $DATA_CONF_PATH --backend awscli

srun /admin/home-knoriy/miniconda3/envs/clasp/bin/python /fsx/knoriy/CLASP/clasp/train.py fit\
    --config $BASE_CONF_PATH \
    --trainer $TRAINER_BASE_CONF_PATH \
    --trainer $TRAINER_CONF_PATH \
    --model $MODEL_CONF_PATH \
    --data $DATA_CONF_PATH \
    --trainer.num_nodes $SLURM_JOB_NUM_NODES \
    --data.num_workers 6 \
    --data.batch_size 16 \
    --trainer.logger.name $LOGGER_NAME \
    --trainer.max_epochs 120 \

# srun sh /fsx/knoriy/CLASP/scripts/clean_tmp_files.sh
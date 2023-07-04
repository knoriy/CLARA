#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --job-name=laion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --account laion
#SBATCH --output=logs/outs/%x_%j.out
#SBATCH --signal=SIGUSR1@90
#SBATCH --signal=SIGTERM@90

module load openmpi

/admin/home-knoriy/miniconda3/envs/clasp/bin/python /fsx/knoriy/CLASP/scripts/download_from_s3.py

srun /admin/home-knoriy/miniconda3/envs/clasp/bin/python /fsx/knoriy/CLASP/clasp/train.py fit\
    --config ./config/config/base.yaml \
    --trainer ./config/config/trainer/base.yaml \
    --trainer ./config/config/trainer/slurm.yaml \
    --model ./config/config/model/pl_clasp.yaml \
    --data ./config/config/data/base.yaml \
    --trainer.num_nodes $SLURM_JOB_NUM_NODES \
    --data.num_workers 12 \
    --data.batch_size 16 \
    --trainer.logger.name large \
    --trainer.max_epochs 1 \

bash scripts/clean_tmp_files.sh
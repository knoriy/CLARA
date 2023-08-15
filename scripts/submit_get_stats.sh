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

srun /admin/home-knoriy/miniconda3/envs/hf/bin/python ./scripts/get_data_stats.py --dataset_name common_voice_11_0 --lang True
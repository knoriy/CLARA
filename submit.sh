#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --account clap
#SBATCH --output=%x_%j.out
#SBATCH --signal=SIGUSR1@90
#SBATCH --signal=SIGTERM@90
#SBATCH --exclude ip-26-0-134-43

module load openmpi
module load cuda/11.7

export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

srun /fsx/home-knoriy/miniconda3/envs/clasp/bin/python /fsx/knoriy/code/CLASP/clasp/train.py \
    --max_epochs 100 \
    --batch_size 32 \
    --accelerator 'gpu' \
    --strategy 'ddp' \
    --num_workers 12 \
    --devices $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) \
    --accumulate_grad_batches 10 \
    --gradient_clip_val 1.0 \
    --logger True \
    --name CLASP_ResNeXt_small_200 \
    --dataset_list /fsx/knoriy/code/CLASP/config/test_list.txt \
    --num_nodes $SLURM_JOB_NUM_NODES \
    # --overfit_batches 1 \
    # --profiler None \ # simple, advanced, pytorch, xla (TPU Only)
    # --checkpoint path/to/checkpoint.pt \
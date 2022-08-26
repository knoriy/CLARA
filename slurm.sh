#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=PL_test
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=6
#SBATCH --comment clap
#SBATCH --output=%x_%j.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
export NCCL_PROTO=simple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib
export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

srun --comment clap /home/knoriy/fsx/miniconda3/envs/clasp/bin/python /home/knoriy/CLASP/project/train.py --max_epochs 1 --accelerator gpu --strategy ddp --devices 8
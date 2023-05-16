#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=base_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --account laion
#SBATCH --output=logs/outs/%x_%j.out
#SBATCH --signal=SIGUSR1@90
#SBATCH --signal=SIGTERM@90
#SBATCH --exclude ip-26-0-134-43,ip-26-0-131-108,ip-26-0-140-150,ip-26-0-143-39

module load openmpi
module load cuda/11.7

export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=1
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

srun /fsx/home-knoriy/miniconda3/envs/clasp_2/bin/python /fsx/knoriy/code/CLASP/clasp/train.py fit\
    --config ./config/config/base.yaml \
    --trainer ./config/config/trainer/base.yaml \
    --trainer ./config/config/trainer/slurm.yaml \
    --model ./config/config/model/pl_clasp.yaml \
    --data ./config/config/data/tensored.yaml \
    --trainer.num_nodes $SLURM_JOB_NUM_NODES \
    --data.num_workers 12 \
    --trainer.logger.name AudioSet_90m_T-Pio \
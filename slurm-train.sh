#!/bin/bash
#SBATCH --partition=g40423
#SBATCH --job-name=CLASP_ResNeXt_small
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --account clap
#SBATCH --output=%x_%j.out
#SBATCH --exclusive

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

echo Running job on $SLURM_JOB_NUM_NODES, 

srun --account clap /fsx/home-knoriy/miniconda3/envs/clasp/bin/python /fsx/knoriy/code/CLASP/clasp/train.py \
    --max_epochs 200 \
    --batch_size 32 \
    --accelerator 'gpu' \
    --strategy 'ddp' \
    --num_workers 6 \
    --devices 8 \
    --num_nodes $SLURM_JOB_NUM_NODES \
    --name $SLURM_JOB_NAME \
    --log_every_n_steps 100000 \
    --accumulate_grad_batches 8 \
    # --precision 16 \
    # --profiler None \ # simple, advanced, pytorch, xla (TPU Only)
    # --checkpoint '/fsx/knoriy/code/CLASP/logs/CLASP/2dkgtlef/checkpoints/epoch=0-step=275000.ckpt' 
    # --checkpoint path/to/checkpoint.pt \
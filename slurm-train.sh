#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=CLASP_ResNeXt_small
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --account clap
#SBATCH --output=%x_%j.out
#SBATCH --signal=SIGUSR1@90

export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export OMPI_MCA_mtl_base_verbose=1
export FI_PROVIDER=efa
export NCCL_TREE_THRESHOLD=0

echo 'Number of Nodes: $($SLURM_JOB_NUM_NODES)'
echo "Number of GPUs available: $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"

srun /fsx/home-knoriy/miniconda3/envs/clasp/bin/python /fsx/knoriy/code/CLASP/clasp/train.py \
    --max_epochs 2 \
    --batch_size 32 \
    --accelerator 'gpu' \
    --strategy 'ddp' \
    --num_workers 6 \
    --devices 8 \
    --log_every_n_steps 10000 \
    --accumulate_grad_batches 8 \
    --gradient_clip_val 1.0 \
    --name $SLURM_JOB_NAME \
    # --num_nodes $SLURM_JOB_NUM_NODES \
    # --precision 16 \
    # --profiler None \ # simple, advanced, pytorch, xla (TPU Only)
    # --checkpoint path/to/checkpoint.pt \
#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=z_out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --account clap
#SBATCH --output=%x_%j.out
#SBATCH --signal=SIGUSR1@90

module load openmpi
module load cuda/11.7

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12910
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

source activate clasp

echo "Number of Nodes: $(echo $SLURM_JOB_NUM_NODES)"
echo "Number of GPUs available: $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"


srun python /fsx/knoriy/code/CLASP/clasp/train.py \
    --max_epochs 200 \
    --batch_size 16 \
    --accelerator 'gpu' \
    --strategy 'ddp' \
    --num_workers 6 \
    --devices 8 \
    --log_every_n_steps 1000 \
    --accumulate_grad_batches 8 \
    --gradient_clip_val 1.0 \
    --logger False \
    --name CLASP_ResNeXt_small_200 \
    --dataset_list /fsx/knoriy/code/CLASP/config/dataset_list.txt \
    --num_nodes $SLURM_JOB_NUM_NODES \
    # --precision 16 \
    # --profiler None \ # simple, advanced, pytorch, xla (TPU Only)
    # --checkpoint path/to/checkpoint.pt \
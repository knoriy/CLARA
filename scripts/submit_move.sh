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

SOURCE_DIR="s3://laion-west/documentaries/"
TARGET_DIR="/data/"
USERNAME="knoriy"
HOST="65.109.157.234"

for file in $(aws s3 ls $SOURCE_DIR --recursive | awk '{print $4}'); do

  # Get S3 file path
  KEY=$(echo $file | sed 's/^.*[[:space:]]//') 
  FILEPATH=${KEY#${SOURCE_DIR}}

  # Remove file name to get just directory path
  DIRPATH=$(dirname "$FILEPATH")

  # Create target directory
  ssh $USERNAME@$HOST "mkdir -p $TARGET_DIR$DIRPATH"


  aws s3 cp s3://laion-west/$file - | ssh $USERNAME@$HOST "cat > $TARGET_DIR$file"
  echo s3://laion-west/$file
done
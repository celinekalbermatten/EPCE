#!/bin/bash
#SBATCH --chdir /scratch/izar/ckalberm
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G
#SBATCH --time 3:00:00

echo STARTING AT `date`
nvidia-smi

cd /home/ckalberm/EPCE
echo SUCCESSFULLY CHANGED LOCATION

python3 -u train.py 

echo FINISHED at `date`
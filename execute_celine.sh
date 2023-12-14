#!/bin/bash
#SBATCH --chdir /scratch/izar/ckalberm
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G

echo STARTING AT `date`
nvidia-smi

cd /home/ckalberm/EPCE
echo SUCCESSFULLY CHANGED LOCATION

python3 -u epce_train_celine_32.py 

echo FINISHED at `date`
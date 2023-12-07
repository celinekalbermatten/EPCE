#!/bin/bash
#SBATCH --chdir /scratch/izar/ckalberm
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --mem 50G

echo STARTING AT `date`

cd /home/ckalberm/EPCE
echo SUCCESSFULLY CHANGED LOCATION

python3 -u epce.py 

echo FINISHED at `date`
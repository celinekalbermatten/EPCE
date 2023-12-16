#!/bin/bash
#SBATCH --chdir /scratch/ckalberm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem 64G
#SBATCH --time 10:00:00

echo STARTING AT `date`

cd /home/ckalberm/EPCE
echo SUCCESSFULLY CHANGED LOCATION

python3 -u train_jed.py 

echo FINISHED at `date`
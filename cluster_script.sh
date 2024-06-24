#!/bin/bash

#Example configuration
#SBATCH -c 6                        # Number of CPU Cores (16)

#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 24G                   # memory pool for all cores
#SBATCH --exclude=loki              # don't use the node loki
#SBATCH --output=slurm.%N.%j.log    # Output and error log (N = node, j = job ID)

#Job commands
export TORCH_HOME=/vol/bitbucket/bc1623/project
source /vol/bitbucket/bc1623/project/uncertainty_env/bin/activate
python /vol/bitbucket/bc1623/project/semi_supervised_uncertainty/train.py
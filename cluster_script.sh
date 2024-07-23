#!/bin/bash
#SBATCH --job-name=vit_mae_chestxray    #  Specify name of job that will appear in the queue
#SBATCH -c 6                        # Number of CPU Cores for this job
#SBATCH -p gpus24                     # Partition (queue), no need to change this
#SBATCH --gres gpu:2                # gpu:n, where n = number of GPUs
#SBATCH --mem 60G                   # Total CPU RAM needed by your job
#SBATCH --output=finetune.%N.%j.log # Standard output and error log
#SBATCH --time=0-12:00:00           # Maximum runtime days-hours:min:sec

#Job commands
export TORCH_HOME=/vol/bitbucket/bc1623/project/semi_supervised_uncertainty
source /vol/bitbucket/bc1623/project/uncertainty_env/bin/activate
python /vol/bitbucket/bc1623/project/semi_supervised_uncertainty/train.py
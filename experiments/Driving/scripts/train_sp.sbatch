#!/bin/bash

# RESOURCES
#SBATCH --nodes=1            # number of nodes to request for job
#SBATCH --ntasks=1           # number if tasks to run
#SBATCH --cpus-per-task=2    # TODO change this, may need more
#SBATCH --mem=16GB           # Total amount of memory required
#SBATCH --time=0-10:00:00    # 0 days, 10 hours, 00 minutes, 00 seconds
#SBATCH --partition=gpu      # slurm partition to use
#SBATCH --gres=gpu:1         # request a single GPU, use gpu:2 for 2 gpus
# OTHER
#SBATCH --output=train_sp_%A_%a.out       # stores stdout/stderr to file
#SBATCH --job-name="train_sp_driving"  # job name used by slurm
#SBATCH --mail-type=all                # send email on job start, end and fault
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au
# ARRAY
#SBATCH --array=0-4          # job array with index values 0, 1, 2, 3, 4

# EXPERIMENT PARAMETERS
python_file=~/code/ba-posgmcp/experiments/Driving/train_sp.py
env_name=Driving7x7RoundAbout-v0
num_iterations=1000

pwd; hostname; date
echo "--- Running Self-Play training for Driving7x7RoundAbout-v0 ---"
echo "Seed=$SLURM_ARRAY_TASK_ID"
singularity exec \
			--nv \
			-B ~/.local-baposgmcp:$Home/.local \
			~/containers/baposgmcp.sif \
			python3 -u $python_file \
			$env_name \
			--num_iterations $num_iterations \
			--seed $SLURM_ARRAY_TASK_ID \
			--num_gpus 1.0 \
			--num_workers 1 \
			--save_policies
echo "Job complete"

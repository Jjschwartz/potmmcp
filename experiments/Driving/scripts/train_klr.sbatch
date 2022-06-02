#!/bin/bash

# Notes on resources for KLR Training
# For nesting_level=k with BR
# need (k+2)*2 cpus
# Each level k >= 0 requires 2 CPUS: 1 learner and 1 rollout worker
# Samer for BR
# Each level requires ~1350 MB GPU RAM, 4000 MB CPU RAM
# So
# --mem=4GB*(k+2)
# --cpus-per-task=2*(k+2)
# --mem-per-gpu=1350*((k+2) / NUM_GPUS)
# --mem-per-cpu=4GB

# Examples
# For k=1, with br and 1 GPU
#  --mem=12GB
#  --cpus-per-task=6
#  --mem-per-gpu=4050MB
#
# For k=2, with br and 1 GPU
#  --mem=16GB
#  --cpus-per-task=8
#  --mem-per-gpu=5400MB
#
# For k=3, with br and 1 GPU
#  --mem=20GB
#  --cpus-per-task=10
#  --mem-per-gpu=6750MB

# Note we can only specify one of:
# --mem
# --mem-per-cpu
# --mem-per-gpu
# So it will be necessary to keep track of --mem-per-gpu to ensure I'm not
# hitting the memory limit for the GPUs available on the server

# RESOURCES
#SBATCH --nodes=1            # number of nodes to request for job
#SBATCH --ntasks=1           # number if tasks to run
#SBATCH --cpus-per-task=14   # TODO change this, with K
#SBATCH --mem=40GB           # TODO change this with K
#SBATCH --time=0-05:00:00    # 0 days, 10 hours, 00 minutes, 00 seconds
#SBATCH --partition=gpu      # slurm partition to use
#SBATCH --gres=gpu:2         # request a single GPU, use gpu:2 for 2 gpus
# OTHER
#SBATCH --output=train_klr_k=4.%J.out      # stores stdout/stderr to file
#SBATCH --job-name="train_klr_k=4"         # job name used by slurm
#SBATCH --mail-type=all                # send email on job start, end and fault
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au
# ARRAY
#SBATCH --array=0-4          # job array with index values 0, 1, 2, 3, 4

# EXPERIMENT PARAMETERS
python_file=~/code/ba-posgmcp/experiments/Driving/train_klr.py
env_name=Driving7x7RoundAbout-v0
nesting_level=4
num_iterations=1000

pwd; hostname; date
echo "--- Running SyKLRBR training for Driving7x7RoundAbout-v0 ---"
echo "Env=$env_name"
echo "K=$nesting_level"
echo "num_iterations=$num_iterations"
echo "Seed=$SLURM_ARRAY_TASK_ID"
singularity exec \
			--nv \
			-B ~/.local-baposgmcp:$Home/.local \
			~/containers/baposgmcp.sif \
			python3 -u $python_file \
			$env_name \
			--k $nesting_level \
			--num_iterations $num_iterations \
			--seed $SLURM_ARRAY_TASK_ID \
			--num_gpus 2.0 \
			--num_workers 1 \
			--train_best_response \
			--save_policies
echo "Job complete"

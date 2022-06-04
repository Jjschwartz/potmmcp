#!/bin/bash

# RESOURCES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    # number of comparisons to be run in parallel
#SBATCH --mem=80GB           # should be > 4GB*cpus-per-task
#SBATCH --time=0-00:03:00    # 0 days, 00 hours, 03 minutes, 00 seconds
# OTHER
#SBATCH --output=pairwise_comparison_test_%J.out
#SBATCH --job-name="pairwise_comparison_test"  # job name used by slurm
#SBATCH --mail-type=all
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au

# EXPERIMENT PARAMETERS
# THIS IS A TEST FILE \0/
python_file=~/code/ba-posgmcp/experiments/Driving/pairwise_comparison.py
env_name=Driving7x7RoundAbout-v0
policy_dirs=~/baposgmcp_results/Driving/rl_policies/2022-06-01_test/*
num_episodes=10

pwd; hostname; date
echo "--- Running Pairwise Comparison for Driving7x7RoundAbout-v0 ---"
echo "Seed=0"
singularity exec \
			-B ~/.local-baposgmcp:$Home/.local \
			~/containers/baposgmcp.sif \
			python3 -u $python_file \
			--env_names $env_name \
			--policy_dirs $policy_dirs \
			--num_episodes $num_episodes \
			--seed 0 \
			--n_procs $SLURM_CPUS_PER_TASK
echo "Job complete"

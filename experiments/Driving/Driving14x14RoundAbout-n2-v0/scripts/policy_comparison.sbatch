#!/bin/bash

# RESOURCES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    # number of comparisons to be run in parallel
#SBATCH --mem=80GB           # should be > 4GB*cpus-per-task
#SBATCH --time=0-24:00:00    # 0 days, 10 hours, 00 minutes, 00 seconds
# OTHER
#SBATCH --output=pairwise_comparison.%J.out
#SBATCH --job-name="pairwise_comparison_driving"  # job name used by slurm
#SBATCH --mail-type=all
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au

# EXPERIMENT PARAMETERS
python_file=~/code/ba-posgmcp/experiments/Driving/pairwise_comparison.py
env_name=Driving14x14WideRoundAbout-v0
policy_dirs=~/baposgmcp_results/Driving/rl_policies/2022-08-13_driving14x14/*
num_episodes=1000

pwd; hostname; date
echo "--- Running Pairwise Comparison for $env ---"
echo "Seed=0"
singularity exec \
			-B ~/.local-baposgmcp:$Home/.local \
			~/containers/baposgmcp.sif \
			python3 -u $python_file \
			--env_names $env_name \
			--policy_dirs $policy_dirs \
			--num_episodes $num_episodes \
			--init_seed 0 \
			--num_seeds 1 \
			--n_procs $SLURM_CPUS_PER_TASK \
			--record_env
echo "Job complete"

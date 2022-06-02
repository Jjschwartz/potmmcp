#!/bin/bash

# RESOURCES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12    # number of comparisons to be run in parallel
#SBATCH --mem=60GB           # should be > 4GB*cpus-per-task
#SBATCH --time=0-24:00:00    # 0 days, 24 hours, 00 minutes, 00 seconds
# OTHER
#SBATCH --output=baposgmcp_experiment.%J.out
#SBATCH --job-name="baposgmcp_exp"  # job name used by slurm
#SBATCH --mail-type=all
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au
# ARRAY
#SBATCH --array=0-4

test_sims=(8 16 32 64)

# EXPERIMENT PARAMETERS
python_file=~/code/ba-posgmcp/experiments/Driving/experiment.py
env_name=Driving7x7RoundAbout-v0
baposgmcp_policy_dirs=~/baposgmcp_results/Driving/rl_policies/*
other_agent_policy_dirs=~/baposgmcp_results/Driving/rl_policies/*
num_episodes=100
time_limit=7200     # 2 hours
num_sims=${test_sims[$SLURM_ARRAY_TASK_ID]}
rollout_policy_ids=(pi_BR pi_SP)


pwd; hostname; date
echo "--- Running BAPOSGMCP Experiment ---"
echo "Env name = $env_name"
echo "BAPOSGMCP Policy dirs = ${baposgmcp_policy_dirs[*]}"
echo "Other agent policy dirs = ${other_agent_policy_dirs[*]}"
echo "Num episodes = $num_episodes"
echo "Time limit = $time_limit"
echo "Num sims = $num_sims"
echo "Rollout policy ids = ${rollout_policy_ids[*]}"
singularity exec \
			-B ~/.local-baposgmcp:$Home/.local \
			~/containers/baposgmcp.sif \
			python3 $python_file \
			--env_names $env_name \
			--baposgmcp_policy_dirs $baposgmcp_policy_dirs \
			--other_agent_policy_dirs $other_agent_policy_dirs \
			--gamma 0.99 \
			--num_episodes $num_episodes \
			--time_limit $time_limit \
			--seed 0 \
			--num_sims $num_sims \
			--rollout_policy_ids ${rollout_policy_ids[*]} \
			--n_procs $SLURM_CPUS_PER_TASK
echo "Job complete"

#!/bin/bash

# RESOURCES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # number of comparisons to be run in parallel
#SBATCH --mem=62GB           # should be > 4GB*cpus-per-task
#SBATCH --time=0-24:00:00    # 0 days, 24 hours, 00 minutes, 00 seconds
# OTHER
#SBATCH --output=baposgmcp_experiment_%J.out
#SBATCH --job-name="baposgmcp_exp"  # job name used by slurm
#SBATCH --mail-type=all
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au
# ARRAY
#SBATCH --array=0-40         # 4 num sims * 10 BAPOSGMCP policy dirs

test_sims=(8 16 32 64)
num_policy_dirs=10
policy_dir_names=( \
train_klr_Driving7x7RoundAbout-v0_k4_seed0_2022-06-01 \
train_klr_Driving7x7RoundAbout-v0_k4_seed1_2022-06-01 \
train_klr_Driving7x7RoundAbout-v0_k4_seed2_2022-06-01 \
train_klr_Driving7x7RoundAbout-v0_k4_seed3_2022-06-01 \
train_klr_Driving7x7RoundAbout-v0_k4_seed4_2022-06-01 \
train_sp_Driving7x7RoundAbout-v0_seed0_2022-06-01 \
train_sp_Driving7x7RoundAbout-v0_seed1_2022-06-01 \
train_sp_Driving7x7RoundAbout-v0_seed2_2022-06-01 \
train_sp_Driving7x7RoundAbout-v0_seed3_2022-06-01 \
train_sp_Driving7x7RoundAbout-v0_seed4_2022-06-01 \
)
base_policy_dir=~/baposgmcp_results/Driving/rl_policies/2022-06-01/

# EXPERIMENT PARAMETERS
python_file=~/code/ba-posgmcp/experiments/Driving/experiment.py
env_name=Driving7x7RoundAbout-v0
baposgmcp_policy_dirs=$base_policy_dir${policy_dir_names[(($SLURM_ARRAY_TASK_ID % $num_policy_dirs))]}
other_agent_policy_dirs=~/baposgmcp_results/Driving/rl_policies/2022-06-01/*
num_episodes=100
time_limit=7200     # 2 hours
num_sims=${test_sims[(($SLURM_ARRAY_TASK_ID/$num_policy_dirs))]}
rollout_policy_ids=(pi_BR pi_SP)

root_save_dir=~/baposgmcp_results/Driving/results/baposgmcp_exp_$(date -I)_$SLURM_JOB_ID
if [ $SLURM_ARRAY_TASK_ID == 0 ]; then
	mkdir -p $root_save_dir
fi

pwd; hostname; date
echo "--- Running BAPOSGMCP Experiment ---"
echo "Env name = $env_name"
echo "BAPOSGMCP Policy dirs = ${baposgmcp_policy_dirs[*]}"
echo "Other agent policy dirs = ${other_agent_policy_dirs[*]}"
echo "Num episodes = $num_episodes"
echo "Time limit = $time_limit"
echo "Num sims = $num_sims"
echo "Rollout policy ids = ${rollout_policy_ids[*]}"
echo "Root save_dir = $root_save_dir"
singularity exec \
			-B ~/.local-baposgmcp:$Home/.local \
			~/containers/baposgmcp.sif \
			python3 -u $python_file \
			--env_names $env_name \
			--baposgmcp_policy_dirs $baposgmcp_policy_dirs \
			--other_agent_policy_dirs $other_agent_policy_dirs \
			--gamma 0.99 \
			--num_episodes $num_episodes \
			--time_limit $time_limit \
			--seed 0 \
			--num_sims $num_sims \
			--rollout_policy_ids ${rollout_policy_ids[*]} \
			--n_procs $SLURM_CPUS_PER_TASK \
			--root_save_dir $root_save_dir
echo "Job complete"

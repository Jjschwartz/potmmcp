#!/bin/bash

# RESOURCES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # number of comparisons to be run in parallel
#SBATCH --mem=62GB           # should be > 4GB*cpus-per-task
# Time = (|other pis| * |seeds|) / cpus-per-task  * time_limit
#      = (8*5)/16 * 8h
#      = 20 h
#SBATCH --time=30:00:00
#SBATCH --partition=gpu      # use GPU partition since more nodes available
# OTHER
#SBATCH --output=baposgmcp_exp_%A_%a.out
#SBATCH --job-name="baposgmcp_exp"  # job name used by slurm
#SBATCH --mail-type=all
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au
# ARRAY
# TODO modify this
# this is to run jobs that didn't run last time :( due to date bug
#SBATCH --array=59-79         # |test sims| * 10 BAPOSGMCP policy dirs

root_save_dir=~/baposgmcp_results/Driving/results/baposgmcp_exp_$SLURM_ARRAY_JOB_ID
if [ $SLURM_ARRAY_TASK_ID == $SLURM_ARRAY_TASK_MIN ]; then
	mkdir -p $root_save_dir
fi

# give each task time for the shared results directory to be created
sleep 2

test_sims=(8 16 32 64 128 256 512 1024)
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
time_limit=28800     # 8 hours
num_sims=${test_sims[(($SLURM_ARRAY_TASK_ID/$num_policy_dirs))]}
rollout_policy_ids=(pi_BR pi_SP)

pwd; hostname; date
echo "--- Running BAPOSGMCP Experiment ---"
echo "Array Job ID = $SLURM_ARRAY_JOB_ID"
echo "Job ID = $SLURM_JOB_ID"
echo "Array task ID = $SLURM_ARRAY_TASK_ID"
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
			--init_seed 0 \
			--num_seeds 1 \
			--num_sims $num_sims \
			--rollout_policy_ids ${rollout_policy_ids[*]} \
			--n_procs $SLURM_CPUS_PER_TASK \
			--root_save_dir $root_save_dir
echo "Job complete"

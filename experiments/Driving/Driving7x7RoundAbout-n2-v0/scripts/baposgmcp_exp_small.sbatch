#!/bin/bash

# RESOURCES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12   # number of comparisons to be run in parallel
# Memory usage
# For 1024 sims need ~20 GB for 10 CPUS (based on past experiments)
#  = 2 GB per CPU
#SBATCH --mem=28GB
# Time = (|other pis| * |pi_seeds| * |exp_seeds|) / cpus-per-task  * time_limit
#      = ciel((1*5*10)/12) * 10h
#      = 50 h
#SBATCH --time=60:00:00
#SBATCH --partition=gpu      # use GPU partition since more nodes available
# OTHER
#SBATCH --output=baposgmcp_exp_%A_%a.out
#SBATCH --job-name="baposgmcp_exp"  # job name used by slurm
#SBATCH --mail-type=all
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au
# ARRAY = |test sims| = 8
#SBATCH --array=0-7

# In this experiment we are testing BAPOSGMCP using a single population of
# policies against a single same other agent policy from all the different
# training populations.
#
# Each experiment run/array job is a different number of simulations
# In total we will run: |test sims| jobs = 8
#
# Within each experiment run/array job we will run:
# BAPOSGMCP vs experiment policy for each different population seed
# |population seeds| = 5
#
# Additionally, for each of pairwise comparison/population seed we run:
# 10 experiment seeds * 100 episodes = 1000 episodes
#
# So total number of individual experiments per array job
#    = |exp seeds| * |population seeds| = 10 * 5 = 50
# Total number of array jobs
#    = |test sims| = 8

root_save_dir=~/baposgmcp_results/Driving/results/baposgmcp_exp_$SLURM_ARRAY_JOB_ID
if [ $SLURM_ARRAY_TASK_ID == $SLURM_ARRAY_TASK_MIN ]; then
	mkdir -p $root_save_dir
fi

# give each task time for the shared results directory to be created
sleep 2

test_sims=(8 16 32 64 128 256 512 1024)
base_policy_dir=~/baposgmcp_results/Driving/rl_policies/2022-07-08_3/
baposgmcp_policy_dir_name=train_klr_Driving7x7RoundAbout-v0_k4_seed0_2022-07-10_13-12-42d6htlanm

# EXPERIMENT PARAMETERS
python_file=~/code/ba-posgmcp/experiments/Driving/experiment.py
env_name=Driving7x7RoundAbout-v0
# use only a single population seed for BAPOSGMCP
baposgmcp_policy_dirs=$base_policy_dir$baposgmcp_policy_dir_name
# test against all population seeds (0-4)
other_agent_policy_dirs=~/baposgmcp_results/Driving/rl_policies/2022-07-08_3/*
init_seed=0
num_seeds=10              # 10 seeds x 100 episodes = 1000 episodes
num_episodes=100
time_limit=36000     # 10 hours
num_sims=${test_sims[(($SLURM_ARRAY_TASK_ID))]}
rollout_policy_ids=pi_BR
# test against a single policy (for each population seed)
include_other_policy_ids=pi_2

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
echo "Init seed = $init_seed"
echo "Num seeds = $num_seeds"
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
			--init_seed $init_seed \
			--num_seeds $num_seeds \
			--num_sims $num_sims \
			--rollout_policy_ids ${rollout_policy_ids[*]} \
			--n_procs $SLURM_CPUS_PER_TASK \
			--root_save_dir $root_save_dir \
			--include_other_policy_ids $include_other_policy_ids \
			--record_env

echo "Job complete"

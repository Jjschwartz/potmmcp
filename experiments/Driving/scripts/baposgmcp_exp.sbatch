#!/bin/bash

# RESOURCES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # number of comparisons to be run in parallel
# Memory usage
# For 1024 sims need ~20 GB for 10 CPUS (based on past experiments)
#  = 2 GB per CPU
#SBATCH --mem=36GB
# Time = (|other pis| * |train seeds|) / cpus-per-task  * time_limit
#      = (7*5)/16 * 12h
#      = ceil(35 exps / 16 exps run at a time) * 8 h
#      = 24 h
#SBATCH --time=30:00:00
#SBATCH --partition=gpu      # use GPU partition since more nodes available
# OTHER
#SBATCH --output=baposgmcp_exp_%A_%a.out
#SBATCH --job-name="baposgmcp_exp"  # job name used by slurm
#SBATCH --mail-type=all
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au
# TODO modify this
# Num jobs = |test sims| * |train seeds|
#          = 4 * 3
#          = 12
#SBATCH --array=0-11

# Each array job is a different (train_seed, num_sims) pair.
# Hence the total number of array jobs is |train_seeds| * |num_sims|
# Which in this run is 3 * 4 = 12

# Within each array job, each experiment

root_save_dir=~/baposgmcp_results/Driving/results/baposgmcp_exp_$SLURM_ARRAY_JOB_ID
if [ $SLURM_ARRAY_TASK_ID == $SLURM_ARRAY_TASK_MIN ]; then
	mkdir -p $root_save_dir
fi

# give each task time for the shared results directory to be created
sleep 2

test_sims=(8 32 128 512)
num_policy_dirs=3
policy_dir_names=( \
train_klr_Driving7x7RoundAbout-v0_k4_seed0_2022-08-03_01-31-491db54_mq \
train_klr_Driving7x7RoundAbout-v0_k4_seed1_2022-08-03_01-33-22b8e8_rdc \
# train_klr_Driving7x7RoundAbout-v0_k4_seed2_2022-08-02_23-21-37dku0mbvb \
train_klr_Driving7x7RoundAbout-v0_k4_seed3_2022-08-02_18-34-07xnsud2q7 \
# train_klr_Driving7x7RoundAbout-v0_k4_seed4_2022-08-02_17-47-19smqri248
)
base_policy_dir=~/baposgmcp_results/Driving/rl_policies/2022-08-02/

# EXPERIMENT PARAMETERS
python_file=~/code/ba-posgmcp/experiments/Driving/experiment.py
env_name=Driving7x7RoundAbout-v0
baposgmcp_policy_dirs=$base_policy_dir${policy_dir_names[(($SLURM_ARRAY_TASK_ID % $num_policy_dirs))]}
other_agent_policy_dirs=~/baposgmcp_results/Driving/rl_policies/2022-08-02/*
time_limit=25200     # 7 hours
num_sims=${test_sims[(($SLURM_ARRAY_TASK_ID/$num_policy_dirs))]}

# the policies BAPOSGMCP will select from for doing rollouts
rollout_policy_ids=(pi_0 pi_1 pi_2 pi_3 pi_4)
# the policies BAPOSGMCP will include in its prior for the other agent
baposgmcp_other_policy_ids=(pi_0 pi_1 pi_2 pi_3)
# the policies that will be run against BAPOSGMCP
exp_other_policy_ids=(pi_-1 pi_0 pi_1 pi_2 pi_3 pi_4 pi_BR)

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
			--init_seed 0 \
			--num_seeds 1 \
			--gamma 0.99 \
			--num_episodes 100 \
			--time_limit $time_limit \
			--num_sims $num_sims \
            --n_procs $SLURM_CPUS_PER_TASK \
		    --record_env \
			--rollout_policy_ids ${rollout_policy_ids[*]} \
			--baposgmcp_other_policy_ids ${baposgmcp_other_policy_ids[*]} \
			--exp_other_policy_ids ${exp_other_policy_ids[*]} \
			--root_save_dir $root_save_dir
echo "Job complete"

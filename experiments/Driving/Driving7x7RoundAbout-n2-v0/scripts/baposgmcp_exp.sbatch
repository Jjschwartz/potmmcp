#!/bin/bash

# RESOURCES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # number of comparisons to be run in parallel
# Memory usage
# For 1024 sims need ~20 GB for 10 CPUS (based on past experiments)
#  = 2 GB per CPU
#SBATCH --mem=32GB       # so it fits within memory limits of nodes
# Time = (|other pis| * |train seeds| * num_seeds) / cpus-per-task  * time_limit
#      = (7*5*4)/16 * 8h
#      = ceil(140 exps / 16 exps run at a time) * 8 h
#      = 72 h
#SBATCH --time=80:00:00
#SBATCH --partition=gpu      # use GPU partition since more nodes available
# OTHER
#SBATCH --output=baposgmcp_exp_%A_%a.out
#SBATCH --job-name="baposgmcp_exp"  # job name used by slurm
#SBATCH --mail-type=all
#SBATCH --mail-user=jonathon.schwartz@anu.edu.au
# TODO modify this
# Num jobs = |test sims| * |train seeds|
#          = 5 * 5
#          = 25
#SBATCH --array=0-24

# Each array job is a different (train_seed, num_sims) pair.
# Hence the total number of array jobs is |train_seeds| * |num_sims|
# Which in this run is 5 * 5 = 25

# Within each array job
# The total number of experiments is
# num_exp_seeds * |exp_other_policy_ids| * |train seeds|
# In this case = 4 * 7 * 5 = 140

root_save_dir=~/baposgmcp_results/Driving/results/baposgmcp_exp_$SLURM_ARRAY_JOB_ID
if [ $SLURM_ARRAY_TASK_ID == $SLURM_ARRAY_TASK_MIN ]; then
	mkdir -p $root_save_dir
fi

# give each task time for the shared results directory to be created
sleep 2

test_sims=(8 32 128 512 1024)
num_policy_dirs=5
policy_dir_names=( \
train_klr_Driving14x14WideRoundAbout-v0_k4_seed0_2022-08-14_14-34-41d5w1geus \
train_klr_Driving14x14WideRoundAbout-v0_k4_seed1_2022-08-14_13-42-26u6tbtyni \
train_klr_Driving14x14WideRoundAbout-v0_k4_seed2_2022-08-14_21-35-000tv77oba \
train_klr_Driving14x14WideRoundAbout-v0_k4_seed3_2022-08-14_21-14-36eb9cg39k \
train_klr_Driving14x14WideRoundAbout-v0_k4_seed4_2022-08-14_19-06-14laiu2uts
)
base_policy_dir=~/baposgmcp_results/Driving/rl_policies/2022-08-13_driving14x14/

# EXPERIMENT PARAMETERS
python_file=~/code/ba-posgmcp/experiments/Driving/experiment.py
env_name=Driving14x14WideRoundAbout-v0
baposgmcp_policy_dirs=$base_policy_dir${policy_dir_names[(($SLURM_ARRAY_TASK_ID % $num_policy_dirs))]}
other_agent_policy_dirs=~/baposgmcp_results/Driving/rl_policies/2022-08-13_driving14x14/*
time_limit=28800     # 8 hours
num_sims=${test_sims[(($SLURM_ARRAY_TASK_ID/$num_policy_dirs))]}

# the policies BAPOSGMCP will select from for doing rollouts
rollout_policy_ids=(pi_0 pi_1 pi_2 pi_3 pi_4)
# the policies BAPOSGMCP will include in its prior for the other agent
baposgmcp_other_policy_ids=(pi_-1 pi_0 pi_1 pi_2 pi_3)
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
			--num_seeds 4 \
			--gamma 0.99 \
			--num_episodes 25 \
			--time_limit $time_limit \
			--num_sims $num_sims \
            --n_procs $SLURM_CPUS_PER_TASK \
		    --record_env \
			--rollout_policy_ids ${rollout_policy_ids[*]} \
			--baposgmcp_other_policy_ids ${baposgmcp_other_policy_ids[*]} \
			--exp_other_policy_ids ${exp_other_policy_ids[*]} \
			--root_save_dir $root_save_dir
echo "Job complete"

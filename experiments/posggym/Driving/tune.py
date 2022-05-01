"""Script for tuning rl policies.

This script doesn't use any of the search or schedule algorithms from ray.tune
but just simply allows for running multiple different experiments in parallel.
"""
import argparse

from ray import tune
from ray.tune.registry import register_env

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy

from baposgmcp import pbt
import baposgmcp.rllib as ba_rllib

from exp_utils import registered_env_creator, RL_TRAINER_CONFIG, get_rllib_env


def _get_config(args):
    config = dict(RL_TRAINER_CONFIG)
    config["env"] = args.env_name
    config["env_config"] = {"env_name": args.env_name, "seed": args.seed}
    config["log_level"] = args.log_level
    config["seed"] = args.seed
    config["num_cpus_per_worker"] = args.num_cpus_per_worker
    config["num_envs_per_worker"] = args.num_envs_per_worker

    sample_env = get_rllib_env(args)
    # obs and action spaces are the same for both agent in TwoPaths env
    obs_space = sample_env.observation_space["0"]
    act_space = sample_env.action_space["0"]

    policy_random_spec = PolicySpec(RandomPolicy, obs_space, act_space, {})
    policy_ppo_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})

    policy_random_id = pbt.get_klr_policy_id(None, -1, True)
    policy_ppo_id = pbt.get_klr_policy_id(None, 0, True)

    policies = {
        policy_random_id: policy_random_spec,
        policy_ppo_id: policy_ppo_spec
    }

    multiagent_config = {
        "policies": policies,
        "policy_mapping_fn": ba_rllib.default_symmetric_policy_mapping_fn,
        "policies_to_train": [policy_ppo_id]
    }
    config["multiagent"] = multiagent_config

    return config


def main(args):
    """Run stuff."""
    # check env name is valid
    get_rllib_env(args)

    register_env(args.env_name, registered_env_creator)

    config = _get_config(args)

    tune.run(
        "PPO",
        name=f"PPO_{args.exp_name}",
        verbose=1,
        num_samples=args.num_samples,
        stop={args.criteria: args.tune_max},
        local_dir=f"~/ray_results/{args.env_name}",
        config=config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env_name", type=str,
        help="Name of the environment to train on."
    )
    parser.add_argument(
        "exp_name", type=str,
        help="Name of the experiment run."
    )
    parser.add_argument(
        "--log_level", type=str, default='WARN',
        help="Log level"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of workers"
    )
    parser.add_argument(
        "--num_gpus", type=float, default=1.0,
        help="Number of GPUs to use."
    )
    parser.add_argument(
        "--num_cpus_per_worker", type=int, default=1,
        help="Number of CPUs per worker."
    )
    parser.add_argument(
        "--num_envs_per_worker", type=int, default=4,
        help="Number of envs per worker."
    )
    parser.add_argument(
        "--num_samples", type=int, default=4,
        help="Number of tune samples to run simoultaneously (?)."
    )
    parser.add_argument(
        "--criteria", type=str, default="timesteps_total",
        help=(
            "Tune time criteria ('training_iteration', 'time_total_s',"
            "timesteps_total)"
        )
    )
    parser.add_argument(
        "--tune_max", type=int, default=50000,
        help="Total tune period, for chosen criteria."
    )
    main(parser.parse_args())

import argparse

import ray
from ray.tune.registry import register_env

import baposgmcp.rllib as ba_rllib

from exp_utils import registered_env_creator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env_name", type=str,
        help="Name of the environment to train on."
    )
    parser.add_argument(
        "policy_dir", type=str,
        help="Path to dir containing trained RL policies"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=2500,
        help="Number of iterations to train."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of worker processes per trainer"
    )
    parser.add_argument(
        "--num_gpus", type=float, default=1.0,
        help="Number of GPUs to use (can be a proportion)."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed."
    )
    parser.add_argument(
        "--save_policies", action="store_true",
        help="Save policies to file at end of training."
    )
    parser.add_argument(
        "--run_serially", action="store_true",
        help="Run training serially."
    )
    args = parser.parse_args()

    ray.init()
    register_env(args.env_name, registered_env_creator)

    ba_rllib.continue_training(
        args.policy_dir,
        is_symmetric=True,
        trainer_class=ba_rllib.BAPOSGMCPPPOTrainer,
        trainers_remote=not args.run_serially,
        num_iterations=args.num_iterations,
        seed=args.seed,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        verbose=True
    )

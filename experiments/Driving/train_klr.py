import argparse

import ray

from ray.tune.registry import register_env

import baposgmcp.rllib as ba_rllib

from exp_utils import registered_env_creator, get_rl_training_config


def main(args):  # noqa
    ray.init()
    register_env(args.env_name, registered_env_creator)

    env = registered_env_creator(
        {"env_name": args.env_name, "seed": args.seed}
    )

    num_trainers = (args.k+1)
    if args.train_best_response:
        num_trainers += 1
    num_gpus_per_trainer = args.num_gpus / num_trainers

    ba_rllib.train_klr_policy(
        args.env_name,
        env,
        k=args.k,
        best_response=args.train_best_response,
        is_symmetric=True,
        seed=args.seed,
        trainer_config=get_rl_training_config(
            args.env_name, args.seed, args.log_level
        ),
        num_workers=args.num_workers,
        num_gpus_per_trainer=num_gpus_per_trainer,
        num_iterations=args.num_iterations,
        run_serially=args.run_serially,
        save_policy=args.save_policy,
        verbose=True
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
        "-k", "--k", type=int, default=3,
        help="Number of reasoning levels"
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
        "--log_level", type=str, default='WARN',
        help="Log level"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed."
    )
    parser.add_argument(
        "--num_gpus", type=float, default=1.0,
        help="Number of GPUs to use (can be a proportion)."
    )
    parser.add_argument(
        "-br", "--train_best_response", action="store_true",
        help="Train a best response on top of KLR policies."
    )
    parser.add_argument(
        "--save_policies", action="store_true",
        help="Save policies to file at end of training."
    )
    parser.add_argument(
        "--run_serially", action="store_true",
        help="Run training serially."
    )
    main(parser.parse_args())

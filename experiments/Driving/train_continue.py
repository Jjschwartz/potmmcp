import argparse
import os.path as osp

import ray
from ray.tune.registry import register_env

import baposgmcp.rllib as ba_rllib

from exp_utils import (
    get_rllib_env,
    registered_env_creator,
    EXP_RL_POLICY_DIR
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

    # check env name is valid
    get_rllib_env(args)

    ray.init()
    register_env(args.env_name, registered_env_creator)

    trainer_args = ba_rllib.TrainerImportArgs(
        trainer_class=ba_rllib.BAPOSGMCPPPOTrainer,
        trainer_remote=not args.run_serially,
        num_workers=args.num_workers,
    )

    igraph, trainers = ba_rllib.import_igraph_trainers(
        igraph_dir=args.policy_dir,
        env_is_symmetric=True,
        trainer_args=trainer_args,
        policy_mapping_fn=None,
        extra_config={},
        seed=args.seed,
        num_gpus=args.num_gpus
    )
    igraph.display()

    ba_rllib.run_training(trainers, igraph, args.num_iterations, verbose=True)

    if args.save_policies:
        print("== Exporting Graph ==")
        policy_dir_name = osp.basename(osp.normpath(args.policy_dir))
        save_dir = "_".join(policy_dir_name.split("_")[:-2])
        export_dir = ba_rllib.export_trainers_to_file(
            EXP_RL_POLICY_DIR,
            igraph,
            trainers,
            trainers_remote=not args.run_serially,
            save_dir_name=save_dir
        )
        print(f"{export_dir=}")

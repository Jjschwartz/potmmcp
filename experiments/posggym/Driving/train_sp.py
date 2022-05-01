import argparse

import ray

from ray.tune.registry import register_env

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from baposgmcp import pbt
import baposgmcp.rllib as ba_rllib

from exp_utils import (
    registered_env_creator, EXP_RL_POLICY_DIR, RL_TRAINER_CONFIG, get_rllib_env
)


def _get_igraph(args) -> pbt.InteractionGraph:
    sample_env = get_rllib_env(args)
    agent_ids = list(sample_env.get_agent_ids())
    agent_ids.sort()

    igraph = pbt.construct_sp_interaction_graph(
        agent_ids, is_symmetric=True, seed=args.seed
    )
    return igraph


def _get_trainer_config(args):
    default_trainer_config = dict(RL_TRAINER_CONFIG)
    default_trainer_config["log_level"] = args.log_level
    default_trainer_config["seed"] = args.seed
    default_trainer_config["env_config"] = {
        "env_name": args.env_name,
        "seed": args.seed
    }

    return {
        "default_trainer_config": default_trainer_config,
        "num_workers": args.num_workers,
        "num_gpus_per_trainer": args.num_gpus
    }


def _get_trainers(args, igraph, trainer_config):
    sample_env = get_rllib_env(args)
    # obs and action spaces are the same for both agent in TwoPaths env
    obs_space = sample_env.observation_space["0"]
    act_space = sample_env.action_space["0"]

    policy_mapping_fn = ba_rllib.get_igraph_policy_mapping_fn(igraph)

    trainers = {}
    train_policy_id = igraph.get_agent_policy_ids(None)[0]
    train_policy_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})
    policy_spec_map = {train_policy_id: train_policy_spec}

    trainer = ba_rllib.get_remote_trainer(
        args.env_name,
        trainer_class=PPOTrainer,
        policies=policy_spec_map,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=[train_policy_id],
        **trainer_config
    )

    trainers[train_policy_id] = trainer
    igraph.update_policy(
        None,
        train_policy_id,
        trainer.get_weights.remote(train_policy_id)
    )

    # need to map from agent_id to trainers
    trainer_map = {pbt.InteractionGraph.SYMMETRIC_ID: trainers}
    return trainer_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env_name", type=str,
        help="Name of the environment to train on."
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
        "--save_policies", action="store_true",
        help="Save policies to file."
    )

    args = parser.parse_args()

    # check env name is valid
    get_rllib_env(args)

    ray.init()
    register_env(args.env_name, registered_env_creator)

    igraph = _get_igraph(args)
    igraph.display()

    trainer_config = _get_trainer_config(args)
    trainers = _get_trainers(args, igraph, trainer_config)

    ba_rllib.run_training(trainers, igraph, args.num_iterations, verbose=True)

    if args.save_policies:
        print("== Exporting Graph ==")
        export_dir = ba_rllib.export_trainers_to_file(
            EXP_RL_POLICY_DIR,
            igraph,
            trainers,
            trainers_remote=True,
            save_dir_name=args.env_name
        )
        print(f"{export_dir=}")

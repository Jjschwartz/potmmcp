"""A script training RL policies."""
import os
import os.path as osp
import pathlib
import argparse
import datetime
from itertools import product

import ray

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy

from baposgmcp import pbt
import baposgmcp.rllib as ba_rllib

from exp_utils import (
    ENV_CONFIG, env_creator, policy_mapping_fn, EXP_RL_POLICY_DIR, ENV_NAME
)


NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))

RL_TRAINER_CONFIG = {
    "env_config": ENV_CONFIG,
    "gamma": 0.95,
    "num_workers": 0,
    "num_envs_per_worker": 4,
    "rollout_fragment_length": 10,
    "train_batch_size": 200,
    "metrics_num_episodes_for_smoothing": 200,
    "model": {
        "use_lstm": True,
        "vf_share_layers": True,
    },
    "num_sgd_iter": 6,
    "vf_loss_coeff": 0.01,
    "observation_filter": "MeanStdFilter",
    "num_gpus": NUM_GPUS,
    "framework": "torch",
}


def _get_trainers_and_igraph(args):
    sample_env = env_creator(ENV_CONFIG)
    # obs and action spaces are the same for both agent in TwoPaths env
    obs_space = sample_env.observation_space["0"]
    act_space = sample_env.action_space["0"]

    l0_policy_spec = PolicySpec(RandomPolicy, obs_space, act_space, {})
    k_policy_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})

    agent_ids = list(sample_env.get_agent_ids())
    agent_ids.sort()
    igraph = pbt.construct_klr_interaction_graph(
        agent_ids, args.k, is_symmetric=False
    )
    igraph.display()
    input("Press Any Key to Continue")

    trainers = {i: {} for i in agent_ids}   # type: ignore
    for agent_k_id, k, agent_km1_id in product(
            agent_ids, range(0, args.k+1), agent_ids
    ):
        if agent_k_id == agent_km1_id:
            continue

        policy_km1_id = pbt.get_klr_policy_id(agent_km1_id, k-1, False)
        policy_k_id = pbt.get_klr_policy_id(agent_k_id, k, False)

        policies_k = {    # type: ignore
            policy_km1_id: l0_policy_spec if k == 0 else k_policy_spec,
            policy_k_id: k_policy_spec
        }

        trainer_k_remote = ray.remote(
            num_cpus=args.num_workers,
            num_gpus=NUM_GPUS/(args.k+1),
            memory=None,
            object_store_memory=None,
            resources=None
        )(PPOTrainer)

        trainer_config = dict(RL_TRAINER_CONFIG)
        trainer_config["multiagent"] = {
            "policies": policies_k,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": [policy_k_id],
        }
        trainer_config["log_level"] = args.log_level

        trainer_k = trainer_k_remote.remote(
            env=ENV_NAME,
            config=trainer_config
        )

        trainers[agent_k_id][policy_k_id] = trainer_k
        igraph.update_policy(
            agent_k_id,
            policy_k_id,
            trainer_k.get_weights.remote(policy_k_id)
        )

    return trainers, igraph


def _train_policies(trainers, igraph, args):
    sample_env = env_creator(ENV_CONFIG)
    agent_ids = list(sample_env.get_agent_ids())
    agent_ids.sort()

    for iteration in range(args.stop_iters):
        print(f"== Iteration {iteration} ==")

        result_futures = {i: {} for i in agent_ids}    # type: ignore
        for i, policy_map in trainers.items():
            for policy_k_id, trainer_k in policy_map.items():
                result_futures[i][policy_k_id] = trainer_k.train.remote()

        results = {i: {} for i in agent_ids}          # type: ignore
        for i, policy_map in result_futures.items():
            results[i] = {
                policy_k_id: ray.get(future)
                for policy_k_id, future in result_futures[i].items()
            }

        for i, policy_map in results.items():
            for k in range(args.k+1):
                print(f"-- Agent ID {i}, Level {k} --")
                policy_k_id = pbt.get_klr_policy_id(i, k, False)
                print(pretty_print(policy_map[policy_k_id]))

                igraph.update_policy(
                    i,
                    policy_k_id,
                    trainers[i][policy_k_id].get_weights.remote(policy_k_id)
                )

        # swap weights of opponent policies
        for i, k, j in product(agent_ids, range(args.k+1), agent_ids):
            if i == j:
                continue
            policy_k_id = pbt.get_klr_policy_id(i, k, False)
            _, opp_weights = igraph.sample_policy(i, policy_k_id, j)
            trainers[i][policy_k_id].set_weights.remote(opp_weights)


def _export_policies_to_file(igraph, trainers):
    print("== Exporting Graph ==")
    export_dir = osp.join(EXP_RL_POLICY_DIR, f"{datetime.datetime.now()}")
    try:
        pathlib.Path(export_dir).mkdir(exist_ok=False)
    except FileExistsError:
        # A timestamp clash is already rare so this should do
        export_dir += "_1"
        pathlib.Path(export_dir).mkdir(exist_ok=False)
    print(f"{export_dir=}")

    igraph.export_graph(
        export_dir,
        ba_rllib.get_trainer_export_fn(
            trainers,
            True,
            # remove unpickalable config values
            config_to_remove=[
                "evaluation_config", ["multiagent", "policy_mapping_fn"]
            ]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of reasoning levels"
    )
    parser.add_argument(
        "--stop-iters", type=int, default=150,
        help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=100000,
        help="Number of timesteps to train."
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="Number of worker processes per trainer"
    )
    parser.add_argument(
        "--log_level", type=str, default='WARN',
        help="Log level"
    )
    args = parser.parse_args()

    ray.init()
    register_env(ENV_NAME, env_creator)

    trainers, igraph = _get_trainers_and_igraph(args)
    _train_policies(trainers, igraph, args)
    _export_policies_to_file(igraph, trainers)

    # Test final policies ??
    # print("== Running Evaluation ==")

"""A script for running KLR PBT in the Two Paths env."""
import os
import logging
import argparse
import tempfile
from itertools import product

import ray

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

import posggym
from posggym.wrappers import FlattenObservation
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

from baposgmcp import pbt
from baposgmcp import runner
import baposgmcp.rllib as ba_rllib
import baposgmcp.stats as stats_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as ba_policy_lib

# pylint: disable=[unused-argument]


def _env_creator(config):
    env = posggym.make(config["env_name"])
    env = FlattenObservation(env)
    return RllibMultiAgentEnv(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env_name", type=str,
        help="Name of TwoPaths environment to run"
    )
    parser.add_argument(
        "--k", type=int, default=1,
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

    # check env is valid
    assert args.env_name.startswith("TwoPaths")
    posggym.make(args.env_name)

    env_config = {"env_name": args.env_name}
    sample_env = _env_creator(env_config)
    # obs and action spaces are the same for both agent in TwoPaths env
    obs_space = sample_env.observation_space["0"]
    act_space = sample_env.action_space["0"]

    def _policy_mapping_fn(agent_id, episode, worker, **kwargs):
        for policy_id in episode.policy_map.keys():
            if policy_id.endswith(agent_id):
                return policy_id
        raise AssertionError

    l0_policy_spec = PolicySpec(RandomPolicy, obs_space, act_space, {})
    k_policy_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})

    agent_ids = list(sample_env.get_agent_ids())
    agent_ids.sort()
    igraph = pbt.construct_klr_interaction_graph(
        agent_ids, args.k, is_symmetric=False
    )
    igraph.display()
    input("Press Any Key to Continue")

    ray.init()
    register_env(args.env_name, _env_creator)

    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))

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
            num_gpus=num_gpus/(args.k+1),
            memory=None,
            object_store_memory=None,
            resources=None
        )(PPOTrainer)

        trainer_k = trainer_k_remote.remote(
            env=args.env_name,
            config={
                "env_config": env_config,
                "gamma": 0.9,
                "num_workers": 0,
                "num_envs_per_worker": 4,
                "rollout_fragment_length": 10,
                "train_batch_size": 200,
                "metrics_num_episodes_for_smoothing": 200,
                "multiagent": {
                    "policies": policies_k,
                    "policy_mapping_fn": _policy_mapping_fn,
                    "policies_to_train": [policy_k_id],
                },
                "model": {
                    "use_lstm": True,
                    "vf_share_layers": True,
                },
                "num_sgd_iter": 6,
                "vf_loss_coeff": 0.01,
                "observation_filter": "MeanStdFilter",
                "num_gpus": num_gpus,
                "log_level": args.log_level,
                "framework": "torch",
            },
        )

        trainers[agent_k_id][policy_k_id] = trainer_k
        igraph.update_policy(
            agent_k_id,
            policy_k_id,
            trainer_k.get_weights.remote(policy_k_id)
        )

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

    # Test saving and loading
    print("== Exporting Graph ==")
    export_dir = tempfile.mkdtemp()
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

    print("\n== Importing Graph ==")

    def _trainer_make_fn(config):
        return PPOTrainer(env=args.env_name, config=config)

    new_igraph, new_trainer_map = ba_rllib.import_igraph_trainers(
        igraph_dir=export_dir,
        env_is_symmetric=False,
        trainer_make_fn=_trainer_make_fn,
        trainers_remote=False,
        policy_mapping_fn=_policy_mapping_fn
    )

    new_igraph.display()
    input("Press Any Key to Continue")

    # Test loaded trainers
    print("== Running Evaluation ==")
    results = {i: {} for i in agent_ids}          # type: ignore
    for i, policy_map in new_trainer_map.items():
        results[i] = {
            policy_k_id: trainer.train()
            for policy_k_id, trainer in policy_map.items()
        }

    for i, policy_map in results.items():
        for k in range(args.k+1):
            print(f"-- Agent ID {i}, Level {k} --")
            policy_k_id = pbt.get_klr_policy_id(i, k, False)
            print(pretty_print(policy_map[policy_k_id]))

    print("\n== Importing Policies ==")
    policy_map = ba_rllib.get_policy_from_trainer_map(new_trainer_map)

    rllibpolicy_map = {}
    env_model = sample_env.unwrapped.model
    for agent_id, agent_policy_map in policy_map.items():
        rllibpolicy_map[agent_id] = {}
        for policy_id, policy in agent_policy_map.items():
            if "-1" in policy_id:
                new_policy = ba_policy_lib.RandomPolicy(
                    env_model, agent_id, 0.9
                )
            else:
                obs_space = env_model.obs_spaces[int(agent_id)]
                new_policy = ba_rllib.RllibPolicy(
                    env_model,
                    agent_id,
                    0.9,
                    policy,
                    preprocessor=ba_rllib.get_flatten_preprocesor(obs_space)
                )
            rllibpolicy_map[agent_id][policy_id] = new_policy

    print("\n== Running Experiments ==")
    # Run the different ba_rllib.RllibPolicy policies against each other
    logging.basicConfig(level="INFO", format='%(message)s')
    agent_0_policies = list(rllibpolicy_map["0"].values())
    agent_1_policies = list(rllibpolicy_map["1"].values())
    for policies in product(agent_0_policies, agent_1_policies):
        trackers = stats_lib.get_default_trackers(policies)
        renderers = [
            render_lib.EpisodeRenderer(pause_each_step=False)
        ]
        runner.run_sims(
            sample_env.unwrapped,
            policies,
            trackers,
            renderers,
            run_config=runner.RunConfig(
                seed=0,
                num_episodes=10
            )
        )

import os
import argparse
from typing import Optional, Dict, Any, Callable, Tuple

import ray
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy

from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

from posggym_agents import pbt

from posggym_agents.pbt import InteractionGraph
from posggym_agents.config import BASE_RESULTS_DIR

from posggym_agents.rllib.utils import RllibTrainerMap
from posggym_agents.rllib.trainer import CustomPPOTrainer
from posggym_agents.rllib.trainer import standard_logger_creator
from posggym_agents.rllib.train import run_training, sync_policies
from posggym_agents.rllib.utils import get_igraph_policy_mapping_fn
from posggym_agents.rllib.export_lib import export_trainers_to_file
from posggym_agents.rllib.utils import posggym_registered_env_creator
from posggym_agents.rllib.trainer import get_remote_trainer, get_trainer


def get_train_klr_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default KLR training args."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env_id", type=str,
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
    return parser


def get_klr_igraph(env: RllibMultiAgentEnv,
                   k: int,
                   best_response: bool,
                   seed: Optional[int]) -> InteractionGraph:
    """Get KLR interaction graph."""
    agent_ids = list(env.get_agent_ids())
    agent_ids.sort()

    if best_response:
        igraph = pbt.construct_klrbr_interaction_graph(
            agent_ids,
            k,
            is_symmetric=env.env.model.is_symmetric,
            dist=None,     # uses poisson with lambda=1.0
            seed=seed
        )
    else:
        igraph = pbt.construct_klr_interaction_graph(
            agent_ids, k, is_symmetric=env.env.model.is_symmetric, seed=seed
        )
    return igraph


def get_symmetric_klr_trainer(env_id: str,
                              env: RllibMultiAgentEnv,
                              igraph: InteractionGraph,
                              seed: Optional[int],
                              trainer_config: Dict[str, Any],
                              num_workers: int,
                              num_gpus_per_trainer: float,
                              logger_creator: Optional[Callable] = None,
                              run_serially: bool = False
                              ) -> RllibTrainerMap:
    """Get Rllib trainer for K-Level Reasoning agents in symmetric env."""
    assert igraph.is_symmetric, "Currently only symmetric envs supported."
    # obs and action spaces are the same for all agents for symmetric envs
    obs_space = env.observation_space["0"]
    act_space = env.action_space["0"]

    policy_mapping_fn = get_igraph_policy_mapping_fn(igraph)

    random_policy_spec = PolicySpec(RandomPolicy, obs_space, act_space, {})
    ppo_policy_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})

    trainers = {}
    for train_policy_id in igraph.get_agent_policy_ids(None):
        connected_policies = igraph.get_all_policies(
            None, train_policy_id, None
        )
        if len(connected_policies) == 0:
            # k = -1
            continue

        if logger_creator is None:
            pi_logger_creator = standard_logger_creator(
                env_id, "klr", seed, train_policy_id
            )

        train_policy_spec = ppo_policy_spec
        policy_spec_map = {train_policy_id: train_policy_spec}
        for (policy_j_id, _) in connected_policies:
            _, k = pbt.parse_klr_policy_id(policy_j_id)
            policy_spec_j = random_policy_spec if k == -1 else ppo_policy_spec
            policy_spec_map[policy_j_id] = policy_spec_j

        if run_serially:
            print("Running serially")
            trainer_k = get_trainer(
                env_id,
                trainer_class=CustomPPOTrainer,
                policies=policy_spec_map,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=[train_policy_id],
                default_trainer_config=trainer_config,
                logger_creator=pi_logger_creator
            )
            trainer_k_weights = trainer_k.get_weights([train_policy_id])
        else:
            trainer_k = get_remote_trainer(
                env_id,
                trainer_class=CustomPPOTrainer,
                policies=policy_spec_map,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=[train_policy_id],
                num_workers=num_workers,
                num_gpus_per_trainer=num_gpus_per_trainer,
                default_trainer_config=trainer_config,
                logger_creator=pi_logger_creator
            )
            trainer_k_weights = trainer_k.get_weights.remote([train_policy_id])

        trainers[train_policy_id] = trainer_k
        igraph.update_policy(None, train_policy_id, trainer_k_weights)

    # need to map from agent_id to trainers
    trainer_map = {pbt.InteractionGraph.SYMMETRIC_ID: trainers}
    return trainer_map


def get_asymmetric_klr_trainer(env_id: str,
                               env: RllibMultiAgentEnv,
                               igraph: InteractionGraph,
                               seed: Optional[int],
                               trainer_config: Dict[str, Any],
                               num_workers: int,
                               num_gpus_per_trainer: float,
                               logger_creator: Optional[Callable] = None,
                               run_serially: bool = False
                               ) -> RllibTrainerMap:
    """Get Rllib trainer for K-Level Reasoning agents in asymmetric env."""
    assert not igraph.is_symmetric

    policy_mapping_fn = get_igraph_policy_mapping_fn(igraph)

    agent_ppo_policy_specs = {
        i: PolicySpec(
            PPOTorchPolicy,
            env.observation_space[str(i)],
            env.action_space[str(i)],
            {}
        )
        for i in igraph.get_agent_ids()
    }
    agent_random_policy_specs = {
        i: PolicySpec(
            RandomPolicy,
            env.observation_space[str(i)],
            env.action_space[str(i)],
            {}
        )
        for i in igraph.get_agent_ids()
    }

    # map from agent_id to agent trainer map
    trainer_map = {i: {} for i in igraph.get_agent_ids()}
    for agent_id in igraph.get_agent_ids():
        for train_policy_id in igraph.get_agent_policy_ids(agent_id):
            connected_policy_ids = []
            for j in igraph.get_agent_ids():
                connected_policies = igraph.get_all_policies(
                    agent_id, train_policy_id, j
                )
                connected_policy_ids.extend([c[0] for c in connected_policies])

            if len(connected_policy_ids) == 0:
                # k = -1
                continue

            if logger_creator is None:
                pi_logger_creator = standard_logger_creator(
                    env_id, "klr", seed, train_policy_id
                )

            train_policy_spec = agent_ppo_policy_specs[agent_id]
            policy_spec_map = {train_policy_id: train_policy_spec}
            for policy_j_id in connected_policy_ids:
                j, k = pbt.parse_klr_policy_id(policy_j_id)
                if k == -1:
                    policy_spec_j = agent_random_policy_specs[j]
                else:
                    policy_spec_j = agent_ppo_policy_specs[j]
                policy_spec_map[policy_j_id] = policy_spec_j

            if run_serially:
                print("Running serially")
                trainer_k = get_trainer(
                    env_id,
                    trainer_class=CustomPPOTrainer,
                    policies=policy_spec_map,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=[train_policy_id],
                    default_trainer_config=trainer_config,
                    logger_creator=pi_logger_creator
                )
                trainer_k_weights = trainer_k.get_weights([train_policy_id])
            else:
                trainer_k = get_remote_trainer(
                    env_id,
                    trainer_class=CustomPPOTrainer,
                    policies=policy_spec_map,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=[train_policy_id],
                    num_workers=num_workers,
                    num_gpus_per_trainer=num_gpus_per_trainer,
                    default_trainer_config=trainer_config,
                    logger_creator=pi_logger_creator
                )
                trainer_k_weights = trainer_k.get_weights.remote(
                    [train_policy_id]
                )

            trainer_map[agent_id][train_policy_id] = trainer_k
            igraph.update_policy(agent_id, train_policy_id, trainer_k_weights)

    sync_policies(trainer_map, igraph)
    return trainer_map


def get_klr_igraph_and_trainer(env_id: str,
                               env: RllibMultiAgentEnv,
                               k: int,
                               best_response: bool,
                               seed: Optional[int],
                               trainer_config: Dict[str, Any],
                               num_workers: int,
                               num_gpus_per_trainer: float,
                               logger_creator: Optional[Callable] = None,
                               run_serially: bool = False
                               ) -> Tuple[InteractionGraph, RllibTrainerMap]:
    """Get igraph and trainer for KLR agent."""
    igraph = get_klr_igraph(env, k, best_response, seed)
    if igraph.is_symmetric:
        trainer_map = get_symmetric_klr_trainer(
            env_id,
            env,
            igraph,
            seed,
            trainer_config,
            num_workers,
            num_gpus_per_trainer,
            logger_creator,
            run_serially
        )
    else:
        trainer_map = get_asymmetric_klr_trainer(
            env_id,
            env,
            igraph,
            seed,
            trainer_config,
            num_workers,
            num_gpus_per_trainer,
            logger_creator,
            run_serially
        )
    return igraph, trainer_map


def train_klr_policy(env_id: str,
                     k: int,
                     best_response: bool,
                     seed: Optional[int],
                     trainer_config: Dict[str, Any],
                     num_workers: int,
                     num_gpus: float,
                     num_iterations: int,
                     run_serially: bool = False,
                     save_policies: bool = True,
                     verbose: bool = True
                     ):
    """Run training of KLR policy."""
    assert "env_config" in trainer_config

    ray.init()
    register_env(env_id, posggym_registered_env_creator)
    env = posggym_registered_env_creator(trainer_config["env_config"])

    num_trainers = k+2 if best_response else k+1
    if not env.env.model.is_symmetric:
        # one trainer per K per agent
        num_trainers *= len(env.get_agent_ids())
    num_gpus_per_trainer = num_gpus / num_trainers

    igraph, trainer_map = get_klr_igraph_and_trainer(
        env_id,
        env,
        k,
        best_response,
        seed,
        trainer_config=trainer_config,
        num_workers=num_workers,
        num_gpus_per_trainer=num_gpus_per_trainer,
        logger_creator=None,
        run_serially=run_serially
    )
    igraph.display()

    run_training(trainer_map, igraph, num_iterations, verbose=verbose)

    if save_policies:
        print("== Exporting Graph ==")
        parent_dir = os.path.join(BASE_RESULTS_DIR, env_id, "policies")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        save_dir = f"train_klr_{env_id}_seed{seed}_k{k}"
        if best_response:
            save_dir += "_br"
        export_dir = export_trainers_to_file(
            parent_dir,
            igraph,
            trainer_map,
            trainers_remote=not run_serially,
            save_dir_name=save_dir
        )
        print(f"{export_dir=}")

"""Code for training self-play agents using rllib."""
import os
import argparse
from typing import Optional, Dict, Any, Callable, Tuple

import ray
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTorchPolicy

from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

from posggym_agents import pbt
from posggym_agents.pbt import InteractionGraph
from posggym_agents.config import BASE_RESULTS_DIR

from posggym_agents.rllib.utils import RllibTrainerMap
from posggym_agents.rllib.trainer import CustomPPOTrainer
from posggym_agents.rllib.trainer import get_remote_trainer
from posggym_agents.rllib.trainer import standard_logger_creator
from posggym_agents.rllib.train import run_training, sync_policies
from posggym_agents.rllib.utils import get_igraph_policy_mapping_fn
from posggym_agents.rllib.export_lib import export_trainers_to_file
from posggym_agents.rllib.utils import posggym_registered_env_creator


def get_train_sp_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default SP training args."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env_id", type=str,
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
        "--save_policy", action="store_true",
        help="Save policies to file."
    )
    return parser


def get_sp_igraph(env: RllibMultiAgentEnv,
                  seed: Optional[int]) -> InteractionGraph:
    """Get self-play interaction graph."""
    agent_ids = list(env.get_agent_ids())
    agent_ids.sort()

    igraph = pbt.construct_sp_interaction_graph(
        agent_ids, is_symmetric=env.env.model.is_symmetric, seed=seed
    )
    return igraph


def get_symmetric_sp_trainer(env_id: str,
                             env: RllibMultiAgentEnv,
                             igraph: InteractionGraph,
                             seed: Optional[int],
                             trainer_config: Dict[str, Any],
                             num_workers: int,
                             num_gpus_per_trainer: float,
                             logger_creator: Optional[Callable] = None
                             ) -> RllibTrainerMap:
    """Get Rllib trainer for self-play trained agents in symmetric env."""
    assert igraph.is_symmetric
    # obs and action spaces are the same for all agents for symmetric envs
    obs_space = env.observation_space["0"]
    act_space = env.action_space["0"]

    policy_mapping_fn = get_igraph_policy_mapping_fn(igraph)

    trainers = {}
    train_policy_id = igraph.get_agent_policy_ids(None)[0]
    train_policy_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})
    policy_spec_map = {train_policy_id: train_policy_spec}

    if logger_creator is None:
        logger_creator = standard_logger_creator(
            env_id, "sp", seed, train_policy_id
        )

    trainer = get_remote_trainer(
        env_id,
        trainer_class=CustomPPOTrainer,
        policies=policy_spec_map,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=[train_policy_id],
        num_workers=num_workers,
        num_gpus_per_trainer=num_gpus_per_trainer,
        default_trainer_config=trainer_config,
        logger_creator=logger_creator
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


def get_asymmetric_sp_trainer(env_id: str,
                              env: RllibMultiAgentEnv,
                              igraph: InteractionGraph,
                              seed: Optional[int],
                              trainer_config: Dict[str, Any],
                              num_workers: int,
                              num_gpus_per_trainer: float,
                              logger_creator: Optional[Callable] = None
                              ) -> RllibTrainerMap:
    """Get Rllib trainer for self-play trained agents in asymmetric env."""
    assert not igraph.is_symmetric
    # assumes a single policy for each agent
    assert all(
        len(igraph.get_agent_policy_ids(i)) == 1
        for i in igraph._agent_ids
    )

    policy_mapping_fn = get_igraph_policy_mapping_fn(igraph)

    policy_spec_map = {}
    for i in igraph.get_agent_ids():
        policy_id = igraph.get_agent_policy_ids(i)[0]
        policy_spec = PolicySpec(
            PPOTorchPolicy,
            env.observation_space[str(i)],
            env.action_space[str(i)],
            {}
        )
        policy_spec_map[policy_id] = policy_spec

    # map from agent_id to agent trainer map
    trainer_map = {}
    for agent_id in igraph.get_agent_ids():
        train_policy_id = igraph.get_agent_policy_ids(agent_id)[0]

        if logger_creator is None:
            logger_creator = standard_logger_creator(
                env_id, "sp", seed, train_policy_id
            )

        trainer = get_remote_trainer(
            env_id,
            trainer_class=CustomPPOTrainer,
            policies=policy_spec_map,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[train_policy_id],
            num_workers=num_workers,
            num_gpus_per_trainer=num_gpus_per_trainer,
            default_trainer_config=trainer_config,
            logger_creator=logger_creator
        )

        trainer_map[agent_id] = {train_policy_id: trainer}
        igraph.update_policy(
            agent_id,
            train_policy_id,
            trainer.get_weights.remote(train_policy_id)
        )

    sync_policies(trainer_map, igraph)
    return trainer_map


def get_sp_igraph_and_trainer(env_id: str,
                              env: RllibMultiAgentEnv,
                              seed: Optional[int],
                              trainer_config: Dict[str, Any],
                              num_workers: int,
                              num_gpus_per_trainer: float,
                              logger_creator: Optional[Callable] = None
                              ) -> Tuple[InteractionGraph, RllibTrainerMap]:
    """Get igraph and trainer for self-play agent."""
    igraph = get_sp_igraph(env, seed)
    if igraph.is_symmetric:
        trainer_map = get_symmetric_sp_trainer(
            env_id,
            env,
            igraph,
            seed,
            trainer_config,
            num_workers,
            num_gpus_per_trainer,
            logger_creator
        )
    else:
        trainer_map = get_asymmetric_sp_trainer(
            env_id,
            env,
            igraph,
            seed,
            trainer_config,
            num_workers,
            num_gpus_per_trainer,
            logger_creator
        )
    return igraph, trainer_map


def train_sp_policy(env_id: str,
                    seed: Optional[int],
                    trainer_config: Dict[str, Any],
                    num_workers: int,
                    num_gpus: float,
                    num_iterations: int,
                    save_policy: bool = True,
                    verbose: bool = True):
    """Run training of self-play policy."""
    assert "env_config" in trainer_config

    ray.init()
    register_env(env_id, posggym_registered_env_creator)
    env = posggym_registered_env_creator(trainer_config["env_config"])

    num_gpus_per_trainer = num_gpus
    if not env.env.model.is_symmetric:
        # one trainer per agent so divide up GPUS
        num_gpus_per_trainer = num_gpus / len(env.get_agent_ids())

    igraph, trainer_map = get_sp_igraph_and_trainer(
        env_id,
        env,
        seed,
        trainer_config=trainer_config,
        num_workers=num_workers,
        num_gpus_per_trainer=num_gpus_per_trainer,
        logger_creator=None
    )
    igraph.display()

    run_training(trainer_map, igraph, num_iterations, verbose=verbose)

    if save_policy:
        print("== Exporting Graph ==")
        parent_dir = os.path.join(BASE_RESULTS_DIR, env_id, "policies")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        save_dir = f"train_sp_{env_id}_seed{seed}"
        export_dir = export_trainers_to_file(
            parent_dir,
            igraph,
            trainer_map,
            trainers_remote=True,
            save_dir_name=save_dir
        )
        print(f"{export_dir=}")

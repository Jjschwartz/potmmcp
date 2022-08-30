"""Code for training self-play agents using rllib."""
import os
from typing import Optional, Dict, Any, Callable, Tuple

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTorchPolicy

from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

from baposgmcp import pbt
from baposgmcp.pbt import InteractionGraph
from baposgmcp.config import BASE_RESULTS_DIR

from baposgmcp.rllib.train import run_training
from baposgmcp.rllib.utils import RllibTrainerMap
from baposgmcp.rllib.trainer import get_remote_trainer
from baposgmcp.rllib.trainer import BAPOSGMCPPPOTrainer
from baposgmcp.rllib.trainer import standard_logger_creator
from baposgmcp.rllib.utils import get_igraph_policy_mapping_fn
from baposgmcp.rllib.export_lib import export_trainers_to_file


def get_sp_igraph(env: RllibMultiAgentEnv,
                  seed: Optional[int]) -> InteractionGraph:
    """Get self-play interaction graph."""
    agent_ids = list(env.get_agent_ids())
    agent_ids.sort()

    obs_space = env.observation_space["0"]
    act_space = env.action_space["0"]
    assert all(env.observation_space[i] == obs_space for i in agent_ids), \
        "Self-play only supported for envs with same obs space for all agents."
    assert all(env.action_space[i] == act_space for i in agent_ids), \
        "Self-play only supported for envs with same act space for all agents."

    igraph = pbt.construct_sp_interaction_graph(
        agent_ids, is_symmetric=True, seed=seed
    )
    return igraph


def get_sp_trainer(env_name: str,
                   env: RllibMultiAgentEnv,
                   igraph: InteractionGraph,
                   seed: Optional[int],
                   trainer_config: Dict[str, Any],
                   num_workers: int,
                   num_gpus_per_trainer: float,
                   logger_creator: Optional[Callable] = None
                   ) -> RllibTrainerMap:
    """Get Rllib trainer for self-play trained agents."""
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
            env_name, "sp", seed, train_policy_id
        )

    trainer = get_remote_trainer(
        env_name,
        trainer_class=BAPOSGMCPPPOTrainer,
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


def get_sp_igraph_and_trainer(env_name: str,
                              env: RllibMultiAgentEnv,
                              seed: Optional[int],
                              trainer_config: Dict[str, Any],
                              num_workers: int,
                              num_gpus_per_trainer: float,
                              logger_creator: Optional[Callable] = None
                              ) -> Tuple[InteractionGraph, RllibTrainerMap]:
    """Get igraph and trainer for self-play agent."""
    igraph = get_sp_igraph(env, seed)
    trainer_map = get_sp_trainer(
        env_name,
        env,
        igraph,
        seed,
        trainer_config,
        num_workers,
        num_gpus_per_trainer,
        logger_creator
    )
    return igraph, trainer_map


def train_sp_policy(env_name: str,
                    env: RllibMultiAgentEnv,
                    seed: Optional[int],
                    trainer_config: Dict[str, Any],
                    num_workers: int,
                    num_gpus_per_trainer: float,
                    num_iterations: int,
                    save_policy: bool = True,
                    verbose: bool = True
                    ):
    """Run training of self-play policy.

    Assumes:
    1. ray has been initialized: ray.init
    2. environment has been registered using ray.tune.registry.register_env
    """
    igraph, trainer_map = get_sp_igraph_and_trainer(
        env_name,
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
        parent_dir = os.path.join(BASE_RESULTS_DIR, env_name, "policies")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        save_dir = f"train_sp_{env_name}_seed{seed}"
        export_dir = export_trainers_to_file(
            parent_dir,
            igraph,
            trainer_map,
            trainers_remote=True,
            save_dir_name=save_dir
        )
        print(f"{export_dir=}")

"""Code for training self-play agents using rllib."""
import os
from typing import Optional, Dict, Any, Callable, Tuple

import ray
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy

from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

from baposgmcp import pbt
from baposgmcp.pbt import InteractionGraph
from baposgmcp.config import BASE_RESULTS_DIR

from baposgmcp.rllib.train import run_training
from baposgmcp.rllib.utils import RllibTrainerMap
from baposgmcp.rllib.trainer import BAPOSGMCPPPOTrainer
from baposgmcp.rllib.trainer import standard_logger_creator
from baposgmcp.rllib.utils import get_igraph_policy_mapping_fn
from baposgmcp.rllib.export_lib import export_trainers_to_file
from baposgmcp.rllib.utils import posggym_registered_env_creator
from baposgmcp.rllib.trainer import get_remote_trainer, get_trainer


def get_klr_igraph(env: RllibMultiAgentEnv,
                   k: int,
                   best_response: bool,
                   is_symmetric: bool,
                   seed: Optional[int]) -> InteractionGraph:
    """Get KLR interaction graph."""
    agent_ids = list(env.get_agent_ids())
    agent_ids.sort()

    if best_response:
        igraph = pbt.construct_klrbr_interaction_graph(
            agent_ids,
            k,
            is_symmetric=is_symmetric,
            dist=None,     # uses poisson with lambda=1.0
            seed=seed
        )
    else:
        igraph = pbt.construct_klr_interaction_graph(
            agent_ids, k, is_symmetric=is_symmetric, seed=seed
        )
    return igraph


def get_klr_trainer(env_name: str,
                    env: RllibMultiAgentEnv,
                    igraph: InteractionGraph,
                    seed: Optional[int],
                    trainer_config: Dict[str, Any],
                    num_workers: int,
                    num_gpus_per_trainer: float,
                    logger_creator: Optional[Callable] = None,
                    run_serially: bool = False
                    ) -> RllibTrainerMap:
    """Get Rllib trainer for self-play trained agents."""
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
                env_name, "klr", seed, train_policy_id
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
                env_name,
                trainer_class=BAPOSGMCPPPOTrainer,
                policies=policy_spec_map,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=[train_policy_id],
                default_trainer_config=trainer_config,
                logger_creator=pi_logger_creator
            )
            trainer_k_weights = trainer_k.get_weights([train_policy_id])
        else:
            trainer_k = get_remote_trainer(
                env_name,
                trainer_class=BAPOSGMCPPPOTrainer,
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


def get_klr_igraph_and_trainer(env_name: str,
                               env: RllibMultiAgentEnv,
                               k: int,
                               best_response: bool,
                               is_symmetric: bool,
                               seed: Optional[int],
                               trainer_config: Dict[str, Any],
                               num_workers: int,
                               num_gpus_per_trainer: float,
                               logger_creator: Optional[Callable] = None,
                               run_serially: bool = False
                               ) -> Tuple[InteractionGraph, RllibTrainerMap]:
    """Get igraph and trainer for KLR agent."""
    igraph = get_klr_igraph(env, k, best_response, is_symmetric, seed)
    trainer_map = get_klr_trainer(
        env_name,
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


def train_klr_policy(env_name: str,
                     k: int,
                     best_response: bool,
                     is_symmetric: bool,
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
    register_env(env_name, posggym_registered_env_creator)
    env = posggym_registered_env_creator(trainer_config["env_config"])

    num_trainers = (k+1)
    if best_response:
        num_trainers += 1
    num_gpus_per_trainer = num_gpus / num_trainers

    igraph, trainer_map = get_klr_igraph_and_trainer(
        env_name,
        env,
        k,
        best_response,
        is_symmetric,
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
        parent_dir = os.path.join(BASE_RESULTS_DIR, env_name, "policies")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        save_dir = f"train_klr_{env_name}_seed{seed}_k{k}"
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

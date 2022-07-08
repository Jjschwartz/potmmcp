import os
import os.path as osp
import pickle
from typing import Dict, Callable, Tuple, Optional, Type
from typing import NamedTuple

import ray
from ray import rllib
from ray.rllib.agents.trainer import Trainer

from baposgmcp import pbt
from baposgmcp.parts import AgentID, PolicyID, Policy

from baposgmcp.rllib.trainer import get_remote_trainer, get_trainer
from baposgmcp.rllib.utils import (
    RllibTrainerMap,
    RllibPolicyMap,
    get_igraph_policy_mapping_fn,
    default_asymmetric_policy_mapping_fn,
    default_symmetric_policy_mapping_fn
)
from baposgmcp.rllib.import_export_utils import (
    TRAINER_CONFIG_FILE, nested_update
)


class TrainerImportArgs(NamedTuple):
    """Object for storing arguments needed for importing trainer."""
    trainer_class: Type[Trainer]
    trainer_remote: bool
    num_workers: Optional[int] = None,
    num_gpus_per_trainer: Optional[float] = None
    logger_creator: Optional[Callable] = None


def _validate_trainer_import_args(trainer_args: TrainerImportArgs):
    if trainer_args.trainer_remote:
        assert trainer_args.num_workers is not None
        assert trainer_args.num_gpus_per_trainer is not None


def _import_trainer_config(import_dir: str) -> Dict:
    config_path = osp.join(import_dir, TRAINER_CONFIG_FILE)
    with open(config_path, "rb") as fin:
        return pickle.load(fin)


def import_trainer(trainer_dir: str,
                   trainer_args: TrainerImportArgs,
                   extra_config: Optional[Dict] = None,
                   ) -> Trainer:
    """Import trainer."""
    _validate_trainer_import_args(trainer_args)

    checkpoints = [
        f for f in os.listdir(trainer_dir) if f.startswith("checkpoint")
    ]
    if len(checkpoints) == 0:
        # untrained policy, e.g. a random policy
        return {}

    # In case multiple checkpoints are stored, take the latest one
    # Checkpoints are named as 'checkpoint_{iteration}'
    checkpoints.sort()
    checkpoint_dir_path = osp.join(trainer_dir, checkpoints[-1])

    # Need to filter checkpoint file from the other files saved alongside
    # the checkpoint (theres probably a better way to do this...)
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir_path)
        if (
            osp.isfile(os.path.join(checkpoint_dir_path, f))
            and f.startswith("checkpoint") and "." not in f
        )
    ]

    checkpoint_path = osp.join(
        checkpoint_dir_path, checkpoint_files[-1]
    )

    config = _import_trainer_config(trainer_dir)

    nested_update(config, extra_config)

    if not trainer_args.trainer_remote:
        trainer = get_trainer(
            env_name=config["env_config"]["env_name"],
            trainer_class=trainer_args.trainer_class,
            policies=config["multiagent"]["policies"],
            policy_mapping_fn=config["multiagent"]["policy_mapping_fn"],
            policies_to_train=config["multiagent"]["policies_to_train"],
            default_trainer_config=config,
            logger_creator=trainer_args.logger_creator
        )
        trainer.restore(checkpoint_path)
        return trainer

    # Import Trainer on remote actor
    remote_trainer = get_remote_trainer(
        env_name=config["env_config"]["env_name"],
        trainer_class=trainer_args.trainer_class,
        policies=config["multiagent"]["policies"],
        policy_mapping_fn=config["multiagent"]["policy_mapping_fn"],
        policies_to_train=config["multiagent"]["policies_to_train"],
        num_workers=trainer_args.num_workers,
        num_gpus_per_trainer=trainer_args.num_gpus_per_trainer,
        default_trainer_config=config,
        logger_creator=trainer_args.logger_creator
    )
    ray.get(remote_trainer.restore.remote(checkpoint_path))  # type: ignore

    return remote_trainer


def get_trainer_weights_import_fn(trainer_args: TrainerImportArgs,
                                  extra_config: Dict,
                                  ) -> Tuple[
                                      pbt.PolicyImportFn, RllibTrainerMap
                                  ]:
    """Get function for importing trained policy weights from local directory.

    The function also returns a reference to a trainer map object which is
    populated with Trainer objects as the trainer import function is called.

    The import function:
    1. Creates a new Trainer object
    2. Restores the trainers state from the file in the import dir
    3. Adds trainer to the trainer map
    4. Returns the weights of the policy with given ID
    """
    trainer_map: Dict[AgentID, Dict[PolicyID, Trainer]] = {}

    def import_fn(agent_id: AgentID,
                  policy_id: PolicyID,
                  import_dir: str) -> Policy:
        trainer = import_trainer(
            trainer_dir=import_dir,
            trainer_args=trainer_args,
            extra_config=extra_config
        )

        if trainer == {}:
            # handle save dirs that contain no exported trainer
            # e.g. save dirs for random policy
            return {}

        if agent_id not in trainer_map:
            trainer_map[agent_id] = {}

        if trainer_args.trainer_remote:
            weights = trainer.get_weights.remote(policy_id)   # type: ignore
        else:
            weights = trainer.get_weights(policy_id)

        trainer_map[agent_id][policy_id] = trainer

        return weights

    return import_fn, trainer_map


def import_policy_trainer(policy_id: PolicyID,
                          igraph_dir: str,
                          env_is_symmetric: bool,
                          agent_id: Optional[AgentID],
                          trainer_args: TrainerImportArgs,
                          policy_mapping_fn: Optional[Callable] = None,
                          extra_config: Optional[Dict] = None,
                          ) -> Trainer:
    """Import trainer for given policy."""
    if agent_id is None or env_is_symmetric:
        agent_id = pbt.InteractionGraph.SYMMETRIC_ID

    if policy_mapping_fn is None and env_is_symmetric:
        policy_mapping_fn = default_symmetric_policy_mapping_fn
    elif policy_mapping_fn is None and not env_is_symmetric:
        policy_mapping_fn = default_asymmetric_policy_mapping_fn

    if extra_config is None:
        extra_config = {}

    if "multiagent" not in extra_config:
        extra_config["multiagent"] = {}

    extra_config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

    policy_dir = osp.join(igraph_dir, agent_id, policy_id)

    trainer = import_trainer(
        policy_dir,
        trainer_args=trainer_args,
        extra_config=extra_config,
    )
    return trainer


def import_policy(policy_id: PolicyID,
                  igraph_dir: str,
                  env_is_symmetric: bool,
                  agent_id: Optional[AgentID],
                  trainer_args: TrainerImportArgs,
                  policy_mapping_fn: Optional[Callable] = None,
                  extra_config: Optional[Dict] = None
                  ) -> rllib.policy.policy.Policy:
    """Import trainer for given policy."""
    trainer = import_policy_trainer(
        policy_id=policy_id,
        igraph_dir=igraph_dir,
        env_is_symmetric=env_is_symmetric,
        agent_id=agent_id,
        trainer_args=trainer_args,
        policy_mapping_fn=policy_mapping_fn,
        extra_config=extra_config
    )

    if trainer == {}:
        # in case of non-trained policy, e.g. for random policy
        return {}

    policy = trainer.get_policy(policy_id)

    # release trainer resources to avoid accumulation of background processes
    trainer.stop()

    return policy


def _dummy_trainer_import_fn(agent_id: AgentID,
                             policy_id: PolicyID,
                             import_dir: str) -> Policy:
    return {}


def import_igraph(igraph_dir: str,
                  env_is_symmetric: bool,
                  seed: Optional[int] = None) -> pbt.InteractionGraph:
    """Import Interaction Graph without loading stored policies."""
    igraph = pbt.InteractionGraph(env_is_symmetric, seed=seed)
    igraph.import_graph(igraph_dir, _dummy_trainer_import_fn)
    return igraph


def _handle_remote_resource_allocation(trainer_args: TrainerImportArgs,
                                       num_trainers: int,
                                       num_gpus: int
                                       ) -> TrainerImportArgs:
    num_workers = trainer_args.num_workers
    if num_workers is None:
        num_workers = 1

    num_gpus_per_trainer = trainer_args.num_gpus_per_trainer
    if num_gpus_per_trainer is None:
        num_gpus_per_trainer = num_gpus / num_trainers

    return TrainerImportArgs(
        trainer_class=trainer_args.trainer_class,
        trainer_remote=trainer_args.trainer_remote,
        num_workers=num_workers,
        num_gpus_per_trainer=num_gpus_per_trainer,
        logger_creator=trainer_args.logger_creator
    )


def import_igraph_trainers(igraph_dir: str,
                           env_is_symmetric: bool,
                           trainer_args: TrainerImportArgs,
                           policy_mapping_fn: Optional[Callable],
                           extra_config: Optional[Dict] = None,
                           seed: Optional[int] = None,
                           num_gpus: Optional[float] = None
                           ) -> Tuple[pbt.InteractionGraph, RllibTrainerMap]:
    """Import Rllib trainers from InteractionGraph directory.

    If policy_mapping_fn is None then will use function from
    baposgmcp.rllib.utils.get_igraph_policy_mapping_function.
    """
    igraph = pbt.InteractionGraph(env_is_symmetric, seed=seed)

    if extra_config is None:
        extra_config = {}

    if "multiagent" not in extra_config:
        extra_config["multiagent"] = {}

    if policy_mapping_fn is None:
        extra_config["multiagent"]["policy_mapping_fn"] = None
        # import igraph without actual policy objects so we can generate
        # policy mapping fn
        igraph.import_graph(igraph_dir, _dummy_trainer_import_fn)
        policy_mapping_fn = get_igraph_policy_mapping_fn(igraph)

    extra_config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

    if trainer_args.trainer_remote:
        trained_policies = igraph.get_outgoing_policies()
        num_trainers = sum(len(v) for v in trained_policies.values())
        trainer_args = _handle_remote_resource_allocation(
            trainer_args, num_trainers, num_gpus
        )

    print(trainer_args)
    input()

    import_fn, trainer_map = get_trainer_weights_import_fn(
        trainer_args=trainer_args,
        extra_config=extra_config
    )

    igraph.import_graph(igraph_dir, import_fn)

    return igraph, trainer_map


def import_igraph_policies(igraph_dir: str,
                           env_is_symmetric: bool,
                           trainer_args: TrainerImportArgs,
                           policy_mapping_fn: Optional[Callable],
                           extra_config: Optional[Dict] = None,
                           ) -> Tuple[pbt.InteractionGraph, RllibPolicyMap]:
    """Import rllib.Policy from InteractionGraph directory.

    Assumes trainers are not Remote.
    """
    igraph = import_igraph(igraph_dir, env_is_symmetric)

    if env_is_symmetric:
        agent_ids = [igraph.SYMMETRIC_ID]
    else:
        agent_ids = igraph.get_agent_ids()

    policy_map: RllibPolicyMap = {}
    for agent_id in agent_ids:
        policy_map[agent_id] = {}
        for policy_id in igraph.get_agent_policy_ids(agent_id):
            print(f"{agent_id=} {policy_id=}")
            policy_map[agent_id][policy_id] = import_policy(
                policy_id=policy_id,
                igraph_dir=igraph_dir,
                env_is_symmetric=env_is_symmetric,
                agent_id=agent_id,
                trainer_args=trainer_args,
                policy_mapping_fn=policy_mapping_fn,
                extra_config=extra_config
            )

    return igraph, policy_map

import os
import os.path as osp
import copy
import pickle
import datetime
import tempfile
from typing import Dict, Callable, Tuple, Sequence, Union, Any, Optional

import ray
from ray import rllib
from ray.rllib.agents.trainer import Trainer

from baposgmcp import pbt
from baposgmcp.parts import AgentID, PolicyID, Policy
from baposgmcp.rllib.utils import (
    RllibTrainerMap,
    RllibPolicyMap,
    get_igraph_policy_mapping_fn,
    default_asymmetric_policy_mapping_fn,
    default_symmetric_policy_mapping_fn
)

TRAINER_CONFIG_FILE = "trainer_config.pkl"


def _import_trainer_config(import_dir: str) -> Dict:
    config_path = os.path.join(import_dir, TRAINER_CONFIG_FILE)
    with open(config_path, "rb") as fin:
        return pickle.load(fin)


def _export_trainer_config(export_dir: str, config: Dict):
    config_path = os.path.join(export_dir, TRAINER_CONFIG_FILE)
    with open(config_path, "wb") as fout:
        pickle.dump(config, fout)


def _nested_update(old: Dict, new: Dict):
    for k, v in new.items():
        if k not in old or not isinstance(v, dict):
            old[k] = v
        else:
            # assume old[k] is also a dict
            _nested_update(old[k], v)


def _nested_remove(old: Dict, to_remove: Sequence[Union[Any, Sequence[Any]]]):
    for keys in to_remove:
        # specify tuple/list since a str is also a sequence
        if not isinstance(keys, (tuple, list)):
            del old[keys]
            continue

        sub_old = old
        for k in keys[:-1]:
            sub_old = sub_old[k]
        del sub_old[keys[-1]]


def get_trainer_export_fn(trainer_map: RllibTrainerMap,
                          trainers_remote: bool,
                          config_to_remove: Sequence[Union[str, Sequence[str]]]
                          ) -> pbt.PolicyExportFn:
    """Get function for exporting trained policies to local directory."""

    def export_fn(agent_id: AgentID,
                  policy_id: PolicyID,
                  policy: Policy,
                  export_dir: str):
        if policy_id not in trainer_map[agent_id]:
            # untrained policy, e.g. a random policy
            return

        trainer = trainer_map[agent_id][policy_id]

        if trainers_remote:
            trainer.set_weights.remote(policy)
            ray.get(trainer.save.remote(export_dir))    # type: ignore
            config = ray.get(trainer.get_config.remote())   # type: ignore
        else:
            trainer.set_weights(policy)
            trainer.save(export_dir)
            config = trainer.config

        config = copy.deepcopy(config)

        # this allows removal of unpickalable objects in config
        _nested_remove(config, config_to_remove)

        _export_trainer_config(export_dir, config)

    return export_fn


def import_trainer(trainer_dir: str,
                   trainer_make_fn: Callable[[Dict], Trainer],
                   trainers_remote: bool,
                   extra_config: Optional[Dict] = None) -> Trainer:
    """Import trainer."""
    checkpoints = [
        f for f in os.listdir(trainer_dir) if f.startswith("checkpoint")
    ]

    if len(checkpoints) == 0:
        # untrained policy, e.g. a random policy
        return {}

    # In case multiple checkpoints are stored, take the latest one
    # Checkpoints are named as 'checkpoint_{iteration}'
    checkpoints.sort()
    checkpoint_dir_path = os.path.join(trainer_dir, checkpoints[-1])

    # Need to filter checkpoint file from the other files saved alongside
    # the checkpoint (theres probably a better way to do this...)
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir_path)
        if (
            os.path.isfile(os.path.join(checkpoint_dir_path, f))
            and f.startswith("checkpoint") and "." not in f
        )
    ]

    checkpoint_path = os.path.join(
        checkpoint_dir_path, checkpoint_files[-1]
    )

    config = _import_trainer_config(trainer_dir)

    _nested_update(config, extra_config)

    trainer = trainer_make_fn(config)

    if trainers_remote:
        ray.get(trainer.restore.remote(checkpoint_path))  # type: ignore
    else:
        trainer.restore(checkpoint_path)

    return trainer


def get_trainer_weights_import_fn(trainer_make_fn: Callable[[Dict], Trainer],
                                  trainers_remote: bool,
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
            trainer_make_fn=trainer_make_fn,
            trainers_remote=trainers_remote,
            extra_config=extra_config
        )

        if trainer == {}:
            # handle save dirs that contain no exported trainer
            # e.g. save dirs for random policy
            return {}

        if agent_id not in trainer_map:
            trainer_map[agent_id] = {}

        if trainers_remote:
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
                          trainer_make_fn: Callable[[Dict], Trainer],
                          policy_mapping_fn: Optional[Callable] = None,
                          trainers_remote: bool = False,
                          extra_config: Optional[Dict] = None) -> Trainer:
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
        policy_dir, trainer_make_fn, trainers_remote, extra_config
    )
    return trainer


def import_policy(policy_id: PolicyID,
                  igraph_dir: str,
                  env_is_symmetric: bool,
                  agent_id: Optional[AgentID],
                  trainer_make_fn: Callable[[Dict], Trainer],
                  policy_mapping_fn: Optional[Callable] = None,
                  trainers_remote: bool = False,
                  extra_config: Optional[Dict] = None
                  ) -> rllib.policy.policy.Policy:
    """Import trainer for given policy."""
    trainer = import_policy_trainer(
        policy_id=policy_id,
        igraph_dir=igraph_dir,
        env_is_symmetric=env_is_symmetric,
        agent_id=agent_id,
        trainer_make_fn=trainer_make_fn,
        policy_mapping_fn=policy_mapping_fn,
        trainers_remote=trainers_remote,
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


def import_igraph_trainers(igraph_dir: str,
                           env_is_symmetric: bool,
                           trainer_make_fn: Callable[[Dict], Trainer],
                           trainers_remote: bool,
                           policy_mapping_fn: Optional[Callable],
                           extra_config: Optional[Dict] = None,
                           seed: Optional[int] = None,
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

    import_fn, trainer_map = get_trainer_weights_import_fn(
        trainer_make_fn, trainers_remote, extra_config
    )

    igraph.import_graph(igraph_dir, import_fn)

    return igraph, trainer_map


def get_policy_from_trainer_map(trainer_map: RllibTrainerMap
                                ) -> RllibPolicyMap:
    """Get map of rllib.Policy from map of rllib.Trainer."""
    policy_map = {}
    for i, agent_trainer_map in trainer_map.items():
        policy_map[i] = {}
        for policy_id, trainer in agent_trainer_map.items():
            policy_map[i][policy_id] = trainer.get_policy(policy_id)
    return policy_map


def import_igraph_policies(igraph_dir: str,
                           env_is_symmetric: bool,
                           trainer_make_fn: Callable[[Dict], Trainer],
                           trainers_remote: bool,
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
                trainer_make_fn=trainer_make_fn,
                policy_mapping_fn=policy_mapping_fn,
                trainers_remote=trainers_remote,
                extra_config=extra_config
            )

    return igraph, policy_map


def export_trainers_to_file(parent_dir: str,
                            igraph: pbt.InteractionGraph,
                            trainers: RllibTrainerMap,
                            trainers_remote: bool,
                            save_dir_name: str = "") -> str:
    """Export Rllib trainer objects to file.

    Handles creation of directory to store
    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir_name = f"{save_dir_name}_{timestr}"
    export_dir = tempfile.mkdtemp(prefix=export_dir_name, dir=parent_dir)

    igraph.export_graph(
        export_dir,
        get_trainer_export_fn(
            trainers,
            trainers_remote,
            # remove unpickalable config values
            config_to_remove=[
                "evaluation_config", ["multiagent", "policy_mapping_fn"]
            ]
        )
    )
    return export_dir

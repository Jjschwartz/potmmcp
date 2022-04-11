import os
import copy
import pickle
from typing import Dict, Callable, Tuple, Sequence, Union, Any, Optional

import ray
from ray import rllib
from ray.rllib.agents.trainer import Trainer

from baposgmcp import pbt
from baposgmcp.parts import AgentID, PolicyID, Policy

TRAINER_CONFIG_FILE = "trainer_config.pkl"

RllibTrainerMap = Dict[AgentID, Dict[PolicyID, Trainer]]
RllibPolicyMap = Dict[AgentID, Dict[PolicyID, rllib.policy.policy.Policy]]


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


def get_trainer_import_fn(trainer_make_fn: Callable[[Dict], Trainer],
                          trainers_remote: bool,
                          extra_config: Dict,
                          ) -> Tuple[pbt.PolicyImportFn, RllibTrainerMap]:
    """Get function for importing trained policies from local directory.

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

        checkpoints = [
            f for f in os.listdir(import_dir) if f.startswith("checkpoint")
        ]

        if len(checkpoints) == 0:
            # untrained policy, e.g. a random policy
            return {}

        # In case multiple checkpoints are stored, take the latest one
        # Checkpoints are named as 'checkpoint_{iteration}'
        checkpoints.sort()
        checkpoint_dir_path = os.path.join(import_dir, checkpoints[-1])

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

        if agent_id not in trainer_map:
            trainer_map[agent_id] = {}

        config = _import_trainer_config(import_dir)

        _nested_update(config, extra_config)

        trainer = trainer_make_fn(config)

        if trainers_remote:
            ray.get(trainer.restore.remote(checkpoint_path))  # type: ignore
            weights = trainer.get_weights.remote(policy_id)   # type: ignore
        else:
            trainer.restore(checkpoint_path)
            weights = trainer.get_weights(policy_id)

        trainer_map[agent_id][policy_id] = trainer

        return weights

    return import_fn, trainer_map


def import_igraph_trainers(igraph_dir: str,
                           env_is_symmetric: bool,
                           trainer_make_fn: Callable[[Dict], Trainer],
                           trainers_remote: bool,
                           policy_mapping_fn: Callable,
                           extra_config: Optional[Dict] = None,
                           ) -> Tuple[pbt.InteractionGraph, RllibTrainerMap]:
    """Import Rllib trainers from InteractionGraph directory."""
    igraph = pbt.InteractionGraph(env_is_symmetric)

    if extra_config is None:
        extra_config = {}

    if "multiagent" not in extra_config:
        extra_config["multiagent"] = {}

    extra_config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

    import_fn, trainer_map = get_trainer_import_fn(
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
                           policy_mapping_fn: Callable,
                           extra_config: Optional[Dict] = None,
                           ) -> Tuple[pbt.InteractionGraph, RllibPolicyMap]:
    """Import rllib.Policy from InteractionGraph directory.

    Assumes trainers are not Remote.
    """
    igraph, trainer_map = import_igraph_trainers(
        igraph_dir,
        env_is_symmetric,
        trainer_make_fn,
        trainers_remote,
        policy_mapping_fn,
        extra_config
    )
    policy_map = get_policy_from_trainer_map(trainer_map)
    return igraph, policy_map

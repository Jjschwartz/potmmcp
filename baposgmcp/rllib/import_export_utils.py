from typing import Dict, Sequence, Union, Any

from baposgmcp.rllib.utils import RllibTrainerMap, RllibPolicyMap


TRAINER_CONFIG_FILE = "trainer_config.pkl"


def nested_update(old: Dict, new: Dict):
    """Update existing dict inplace with a new dict, handling nested dicts."""
    for k, v in new.items():
        if k not in old or not isinstance(v, dict):
            old[k] = v
        else:
            # assume old[k] is also a dict
            nested_update(old[k], v)


def nested_remove(old: Dict, to_remove: Sequence[Union[Any, Sequence[Any]]]):
    """Remove items from an existing dict, handling nested sequences."""
    for keys in to_remove:
        # specify tuple/list since a str is also a sequence
        if not isinstance(keys, (tuple, list)):
            del old[keys]
            continue

        sub_old = old
        for k in keys[:-1]:
            sub_old = sub_old[k]
        del sub_old[keys[-1]]


def get_policy_from_trainer_map(trainer_map: RllibTrainerMap
                                ) -> RllibPolicyMap:
    """Get map of rllib.Policy from map of rllib.Trainer."""
    policy_map = {}
    for i, agent_trainer_map in trainer_map.items():
        policy_map[i] = {}
        for policy_id, trainer in agent_trainer_map.items():
            policy_map[i][policy_id] = trainer.get_policy(policy_id)
    return policy_map

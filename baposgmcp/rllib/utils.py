from typing import Callable, Any

from gym import spaces

import posggym.model as M


ObsPreprocessor = Callable[[M.Observation], Any]


def identity_preprocessor(obs: M.Observation) -> Any:
    """Return the observation unchanged."""
    return obs


def get_flatten_preprocesor(obs_space: spaces.Space) -> ObsPreprocessor:
    """Get the preprocessor function for flattening observations."""

    def flatten_preprocessor(obs: M.Observation) -> Any:
        """Flatten the observation."""
        return spaces.flatten(obs_space, obs)

    return flatten_preprocessor


def default_asymmetric_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Get the default policy mapping function for asymmetric environments.

    Assumes the policy ID naming convention used by baposgmcp.pbt.
    """
    for policy_id in episode.policy_map.keys():
        if policy_id.endswith(agent_id):
            return policy_id
    raise AssertionError


def default_symmetric_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Get the default policy mapping function for asymmetric environments.

    Assumes the policy ID naming convention used by baposgmcp.pbt.
    Also assumes agent id is an integer or a str representation of an integer.
    """
    policy_ids = list(episode.policy_map.keys())
    policy_ids.sort()
    return policy_ids[int(agent_id)]

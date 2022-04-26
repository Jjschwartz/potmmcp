import random
from typing import Callable, Any, Dict, List, Union, Optional

from ray import rllib
from ray.rllib.agents.trainer import Trainer

import numpy as np
from gym import spaces

import posggym.model as M

from baposgmcp.parts import AgentID, PolicyID


ObsPreprocessor = Callable[[M.Observation], Any]
RllibTrainerMap = Dict[AgentID, Dict[PolicyID, Trainer]]
RllibPolicyMap = Dict[AgentID, Dict[PolicyID, rllib.policy.policy.Policy]]


def identity_preprocessor(obs: M.Observation) -> Any:
    """Return the observation unchanged."""
    return obs


def get_flatten_preprocessor(obs_space: spaces.Space) -> ObsPreprocessor:
    """Get the preprocessor function for flattening observations."""

    def flatten_preprocessor(obs: M.Observation) -> Any:
        """Flatten the observation."""
        return spaces.flatten(obs_space, obs)

    return flatten_preprocessor


def default_asymmetric_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Get the default policy mapping function for asymmetric environments.

    Assumes the policy ID naming convention used by baposgmcp.pbt and assumes
    there is only one policy per agent.
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


def uniform_asymmetric_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map from agent id to available agent policy uniformly at random.

    Assumes the policy ID naming convention used by baposgmcp.pbt.
    """
    agent_policies = []
    for policy_id in episode.policy_map.keys():
        if policy_id.endswith(agent_id):
            agent_policies.append(policy_id)
    return random.choice(agent_policies)


def get_custom_policy_mapping_fn(agent_policy_map: Dict[
                                     AgentID, Union[PolicyID, List[PolicyID]]
                                 ],
                                 agent_policy_dist: Optional[
                                     Dict[AgentID, List[float]]
                                 ],
                                 seed: Optional[int]) -> Callable:
    """Get a custom Rllib policy mapping function.

    If agent_policy_dist is None then uses a uniform distribution (i.e. chooses
    a policy for a given agent uniformly at random from the list of available
    policies for that agent).
    """
    rng = random.Random(seed)

    for agent_id in agent_policy_map:
        if not isinstance(agent_policy_map[agent_id], list):
            agent_policy_map[agent_id] = [agent_policy_map[agent_id]]

    if agent_policy_dist is None:
        agent_policy_dist = {}
        for agent_id, policies in agent_policy_map.items():
            agent_policy_dist[agent_id] = [1.0 / len(policies)] * len(policies)

    assert all(
        len(agent_policy_map[i]) == len(agent_policy_dist[i])
        for i in agent_policy_map
    )

    def dist_mapping_fn(agent_id, episode, worker, **kwargs):
        agent_policies = agent_policy_map[agent_id]
        policy_weights = agent_policy_dist[agent_id]
        return rng.choices(agent_policies, weights=policy_weights, k=1)[0]

    return dist_mapping_fn


def numpy_softmax(x: np.ndarray) -> np.ndarray:
    """Perform the softmax function on an array."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

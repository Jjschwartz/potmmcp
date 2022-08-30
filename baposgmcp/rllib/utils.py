import random
from typing import Callable, Any, Dict, List, Union, Optional

from ray import rllib
from ray.tune.registry import register_env
from ray.rllib.agents.trainer import Trainer

import numpy as np
from gym import spaces

import posggym
import posggym.model as M
from posggym.wrappers import FlattenObservation
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

from baposgmcp import pbt
from baposgmcp.parts import AgentID
from baposgmcp.policy import PolicyID


ObsPreprocessor = Callable[[M.Observation], Any]
RllibTrainerMap = Dict[AgentID, Dict[PolicyID, Trainer]]
RllibPolicyMap = Dict[AgentID, Dict[PolicyID, rllib.policy.policy.Policy]]


def posggym_registered_env_creator(config):
    """Create a new rllib compatible environment from POSGgym environment.

    Config expects:
    "env_name" - name of the posggym env
    "seed" - environment seed
    "flatten_obs" - bool whether to use observation flattening wrapper
                   (default=True)
    """
    env = posggym.make(config["env_name"], **{"seed": config["seed"]})
    if config.get("flatten_obs", True):
        env = FlattenObservation(env)
    return RllibMultiAgentEnv(env)


def register_posggym_env(env_name: str):
    """Register posggym env with Ray."""
    print(f"\n\n\nRegistered {env_name=} \n\n\n")
    register_env(env_name, posggym_registered_env_creator)


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

    Assumes:
    - the policy ID naming convention used by baposgmcp.pbt
    - there is only one policy per agent.
    """
    for policy_id in episode.policy_map.keys():
        if policy_id.endswith(agent_id):
            return policy_id
    raise AssertionError


def get_igraph_policy_mapping_fn(igraph: pbt.InteractionGraph) -> Callable:
    """Get Policy mapping fn from interactive graph.

    May not work as expected for envs with more than 2 agents when an Rllib
    trainer has more than a single policy to train.

    Double check before using this function in those cases.
    """

    def symmetric_mapping_fn(agent_id, episode, worker, **kwargs):
        policies_to_train = [
            pid for pid in worker.policy_map.keys()
            if worker.is_policy_to_train(pid, None)
        ]
        policies_to_train.sort()

        # episode_id is selected using random.randrange(2e9)
        # so we use it to choose the policy to train from the set of policies
        train_policy_idx = episode.episode_id % len(policies_to_train)
        train_policy_id = policies_to_train[train_policy_idx]

        # since env is symmetric it doesn't matter which agent id is the train
        # agent, so we just set it to always be the first agent id
        if agent_id == igraph.get_agent_ids()[0]:
            return train_policy_id
        return igraph.sample_policy(None, train_policy_id, None)[0]

    def asymmetric_mapping_fn(agent_id, episode, worker, **kwargs):
        policies_to_train = [
            pid for pid in worker.policy_map.keys()
            if worker.is_policy_to_train(pid, None)
        ]
        policies_to_train.sort()

        # episode_id is selected using random.randrange(2e9)
        # so we use it to choose the policy to train from the set of policies
        train_policy_idx = episode.episode_id % len(policies_to_train)
        train_policy_id = policies_to_train[train_policy_idx]

        train_agent_id = pbt.get_agent_id_from_policy_id(train_policy_id)
        if agent_id == train_agent_id:
            return train_policy_id

        return igraph.sample_policy(
            train_agent_id, train_policy_id, agent_id
        )

    if igraph.is_symmetric:
        return symmetric_mapping_fn
    return asymmetric_mapping_fn


def default_symmetric_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Get the default policy mapping function for asymmetric environments.

    Assumes:
    - the policy ID naming convention used by baposgmcp.pbt.
    - agent id is an integer or a str representation of an integer.
    - there is only a single policy for each agent in the environment.
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


def get_custom_asymetric_policy_mapping_fn(agent_policy_map: Dict[
                                              AgentID,
                                              Union[PolicyID, List[PolicyID]]
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


def get_asymmetric_br_policy_mapping_fn(policy_br_id: str,
                                        br_agent_id: str,
                                        other_policy_ids: List[str],
                                        other_policy_dist: List[float],
                                        seed: Optional[int]):
    """Get policy mapping fn for BR agent in an asymmetric env."""
    rng = random.Random(seed)

    def mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == br_agent_id:
            return policy_br_id
        return rng.choices(
            other_policy_ids, weights=other_policy_dist, k=1
        )[0]

    return mapping_fn


def get_symmetric_br_policy_mapping_fn(policy_br_id: str,
                                       other_policy_ids: List[str],
                                       other_policy_dist: List[float],
                                       seed: Optional[int]):
    """Get policy mapping fn for BR agent in a symmetric env."""
    rng = random.Random(seed)

    def mapping_fn(agent_id, episode, worker, **kwargs):
        episode_agent_ids = episode.get_agents()
        episode_agent_ids.sort()
        if agent_id == episode_agent_ids[0]:
            return policy_br_id
        return rng.choices(
            other_policy_ids, weights=other_policy_dist, k=1
        )[0]

    return mapping_fn


def numpy_softmax(x: np.ndarray) -> np.ndarray:
    """Perform the softmax function on an array."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

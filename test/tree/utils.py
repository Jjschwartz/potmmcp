import logging
from typing import Optional
from collections import defaultdict

import posggym.model as M
from posggym.utils.history import AgentHistory

from posggym_agents.agents.random import RandomPolicy
from posggym_agents.agents.random import FixedDistributionPolicy
from posggym_agents.policy import PolicyHiddenState

import potmmcp.run as run_lib
import potmmcp.tree as tree_lib
from potmmcp.meta_policy import DictMetaPolicy
from potmmcp.policy_prior import UniformPolicyPrior


class RandomPolicyWithValue(RandomPolicy):
    """Uniform Random Policy with fixed value estimate."""

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: M.AgentID,
        policy_id: str,
        value: float = 1.0,
    ):
        super().__init__(
            model,
            agent_id,
            policy_id,
        )
        self.value = value

    def get_value(self, history: Optional[AgentHistory]) -> float:
        return self.value

    def get_value_by_hidden_state(self, hidden_state: PolicyHiddenState) -> float:
        return self.value


def run_sims(env, policies, num_episodes, trackers, render=False, **kwargs):  # noqa
    logging.basicConfig(level=logging.INFO - 2, format="%(message)s")

    renderers = []
    if render:
        renderers.append(run_lib.EpisodeRenderer())

    run_lib.run_episodes(env, policies, num_episodes, trackers, renderers, **kwargs)


def get_deterministic_policies(env, agent_id):  # noqa
    policies = {}
    for pi_a in range(env.action_spaces[agent_id].n):
        dist = {a: 0.0 for a in range(env.action_spaces[agent_id].n)}
        dist[pi_a] = 1.0
        policies[f"pi_{pi_a}"] = FixedDistributionPolicy(
            env.model, agent_id=agent_id, policy_id=f"pi_{pi_a}", dist=dist
        )
    return policies


def get_biased_policies(env, agent_id, bias):  # noqa
    policies = {}
    n_actions = env.action_spaces[agent_id].n
    p_bias = (1.0 / n_actions) + bias
    p_non_biased = (1.0 / n_actions) - (bias / (n_actions - 1))
    for pi_a in range(n_actions):
        dist = {a: p_non_biased for a in range(n_actions)}
        dist[pi_a] = p_bias
        policies[f"pi_{pi_a}"] = FixedDistributionPolicy(
            env.model, agent_id=agent_id, policy_id=f"pi_{pi_a}", dist=dist
        )
    return policies


def get_random_policy(env, agent_id, value=1.0):  # noqa
    return RandomPolicyWithValue(env.model, agent_id, "pi_-1", value)


def get_deterministic_other_policies(env, ego_agent):  # noqa
    return {
        i: get_deterministic_policies(env, i)
        for i in range(env.n_agents)
        if i != ego_agent
    }


def get_deterministic_other_policy_prior(env, ego_agent):  # noqa
    policies = get_deterministic_other_policies(env, ego_agent)
    return UniformPolicyPrior(env.model, ego_agent, policies)


def get_biased_other_policies(env, ego_agent, bias):  # noqa
    return {
        i: get_biased_policies(env, i, bias)
        for i in range(env.n_agents)
        if i != ego_agent
    }


def get_biased_other_policy_prior(env, ego_agent, bias):  # noqa
    policies = get_biased_other_policies(env, ego_agent, bias)
    return UniformPolicyPrior(env.model, ego_agent, policies)


def get_random_other_policies(env, ego_agent):  # noqa
    return {
        i: {"pi_-1": get_random_policy(env, i)}
        for i in range(env.n_agents)
        if i != ego_agent
    }


def get_random_other_policy_prior(env, ego_agent):  # noqa
    policies = get_random_other_policies(env, ego_agent)
    return UniformPolicyPrior(env.model, ego_agent, policies)


def get_random_meta_policy(env, ego_agent, value=1.0):  # noqa
    ego_policies = {"pi_-1": get_random_policy(env, ego_agent, value)}
    meta_policy_dict = defaultdict(lambda: {"pi_-1": 1.0})
    return DictMetaPolicy(env.model, ego_agent, ego_policies, meta_policy_dict)


def get_random_potmmcp(
    env,
    ego_agent,
    other_policy_prior,
    meta_policy,
    truncated,
    step_limit,
    num_sims=64,
    value=1.0,
):  # noqa
    if other_policy_prior is None:
        other_policy_prior = get_random_other_policy_prior(env, ego_agent)

    if meta_policy is None:
        meta_policy = get_random_meta_policy(env, ego_agent, value)

    return tree_lib.POTMMCP(
        env.model,
        agent_id=ego_agent,
        discount=0.9,
        num_sims=num_sims,
        search_time_limit=None,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        c=1.25,
        truncated=truncated,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
        step_limit=step_limit,
    )

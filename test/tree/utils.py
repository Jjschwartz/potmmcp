import logging
from collections import defaultdict

import baposgmcp.policy as P
import baposgmcp.run as run_lib
import baposgmcp.tree as tree_lib
from baposgmcp.meta_policy import DictMetaPolicy
from baposgmcp.policy_prior import UniformPolicyPrior


def run_sims(env, policies, trackers, run_config, render=False):    # noqa
    logging.basicConfig(level=logging.INFO-2, format='%(message)s')

    renderers = []
    if render:
        renderers.append(run_lib.EpisodeRenderer())

    run_lib.run_sims(env, policies, trackers, renderers, run_config)


def get_deterministic_policies(env, agent_id):  # noqa
    policies = {}
    for pi_a in range(env.action_spaces[agent_id].n):
        dist = {a: 0.0 for a in range(env.action_spaces[agent_id].n)}
        dist[pi_a] = 1.0
        policies[f"pi_{pi_a}"] = P.FixedDistributionPolicy(
            env.model,
            ego_agent=agent_id,
            gamma=0.9,
            dist=dist,
            policy_id=f"pi_{pi_a}"
        )
    return policies


def get_biased_policies(env, agent_id, bias):   # noqa
    policies = {}
    n_actions = env.action_spaces[agent_id].n
    p_bias = (1.0 / n_actions) + bias
    p_non_biased = (1.0 / n_actions) - (bias / (n_actions-1))
    for pi_a in range(n_actions):
        dist = {a: p_non_biased for a in range(n_actions)}
        dist[pi_a] = p_bias
        policies[f"pi_{pi_a}"] = P.FixedDistributionPolicy(
            env.model,
            ego_agent=agent_id,
            gamma=0.9,
            dist=dist,
            policy_id=f"pi_{pi_a}"
        )
    return policies


def get_random_policy(env, agent_id):   # noqa
    return P.RandomPolicy(env.model, agent_id, 0.9, "pi_-1")


def get_deterministic_other_policies(env, ego_agent):  # noqa
    return {
        i: get_deterministic_policies(env, i)
        for i in range(env.n_agents) if i != ego_agent
    }

def get_deterministic_other_policy_prior(env, ego_agent):  # noqa
    policies = get_deterministic_other_policies(env, ego_agent)
    return UniformPolicyPrior(env.model, ego_agent, policies)


def get_biased_other_policies(env, ego_agent, bias):  # noqa
    return {
        i: get_biased_policies(env, i, bias)
        for i in range(env.n_agents) if i != ego_agent
    }


def get_biased_other_policy_prior(env, ego_agent, bias):  # noqa
    policies = get_biased_other_policies(env, ego_agent, bias)
    return UniformPolicyPrior(env.model, ego_agent, policies)


def get_random_other_policies(env, ego_agent):   # noqa
    return {
        i: {"pi_-1": P.RandomPolicy(env.model, i, 0.9, "pi_-1")}
        for i in range(env.n_agents) if i != ego_agent
    }


def get_random_other_policy_prior(env, ego_agent):   # noqa
    policies = get_random_other_policies(env, ego_agent)
    return UniformPolicyPrior(env.model, ego_agent, policies)


def get_random_meta_policy(env, ego_agent):  # noqa
    ego_policies = {
        'pi_-1': P.RandomPolicy(env.model, ego_agent, 0.9, "pi_-1")
    }
    meta_policy_dict = defaultdict(lambda: {'pi_-1': 1.0})
    return DictMetaPolicy(
        env.model, ego_agent, ego_policies, meta_policy_dict
    )


def get_random_baposgmcp(env,
                         ego_agent,
                         other_policy_prior,
                         meta_policy,
                         truncated,
                         step_limit,
                         num_sims=64):   # noqa
    if other_policy_prior is None:
        other_policy_prior = get_random_other_policy_prior(env, ego_agent)

    if meta_policy is None:
        meta_policy = get_random_meta_policy(env, ego_agent)

    return tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=ego_agent,
        gamma=0.9,
        num_sims=num_sims,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        c_init=1.0,
        c_base=100.0,
        truncated=truncated,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
        step_limit=step_limit

    )

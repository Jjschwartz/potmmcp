import logging

import baposgmcp.policy as P
import baposgmcp.run as run_lib
import baposgmcp.tree as tree_lib


def run_sims(env, policies, trackers, run_config, render=False):    # noqa
    logging.basicConfig(level=logging.DEBUG-2, format='%(message)s')

    renderers = []
    if render:
        renderers.append(run_lib.EpisodeRenderer())

    run_lib.run_sims(env, policies, trackers, renderers, run_config)


def get_rps_deterministic_policies(env, agent_id):  # noqa
    other_agent_policies = {agent_id: {}}
    for pi_a in range(env.action_spaces[agent_id].n):
        dist = {a: 0.0 for a in range(env.action_spaces[agent_id].n)}
        dist[pi_a] = 1.0
        pi = P.FixedDistributionPolicy(
            env.model,
            ego_agent=agent_id,
            gamma=0.9,
            dist=dist,
            policy_id=f"pi_{pi_a}"
        )
        other_agent_policies[agent_id][f"pi_{pi_a}"] = pi
    return other_agent_policies


def get_rps_biased_policies(env, agent_id, bias):   # noqa
    other_agent_policies = {agent_id: {}}
    p_non_biased = (1.0 - bias) / 2
    for pi_a in range(env.action_spaces[agent_id].n):
        dist = {a: p_non_biased for a in range(env.action_spaces[agent_id].n)}
        dist[pi_a] = bias
        pi = P.FixedDistributionPolicy(
            env.model,
            ego_agent=agent_id,
            gamma=0.9,
            dist=dist,
            policy_id=f"pi_{pi_a}"
        )
        other_agent_policies[agent_id][f"pi_{pi_a}"] = pi
    return other_agent_policies


def get_rps_random_policy(env, agent_id):   # noqa
    return {agent_id: {
        'pi_-1': P.RandomPolicy(env.model, agent_id, 0.9, "pi_-1")
    }}


def get_random_rollout_policy(env, ego_agent, other_policies):  # noqa
    rollout_policies = {
        'pi_-1': P.RandomPolicy(env.model, ego_agent, 0.9, "pi_-1")
    }
    rollout_selection = {
        pi_id: 'pi_-1' for pi_id in other_policies
    }
    return rollout_policies, rollout_selection


def get_random_baposgmcp(env,
                         ego_agent,
                         other_policies,
                         truncated,
                         step_limit):   # noqa
    rollout_policies, rollout_selection = get_random_rollout_policy(
        env, ego_agent, other_policies[(ego_agent + 1) % 2]
    )
    return tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=ego_agent,
        gamma=0.9,
        num_sims=64,
        other_policies=other_policies,
        other_policy_prior=None,
        rollout_policies=rollout_policies,
        rollout_selection=rollout_selection,
        c_init=1.0,
        c_base=100.0,
        truncated=truncated,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
        step_limit=step_limit

    )

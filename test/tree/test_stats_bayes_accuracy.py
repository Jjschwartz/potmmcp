import logging

import posggym

import baposgmcp.run as run_lib
import baposgmcp.tree as tree_lib
import baposgmcp.policy as policy_lib

RENDER = False


def _run_sims(env, policies, trackers, run_config):
    logging.basicConfig(level=logging.DEBUG-2, format='%(message)s')

    renderers = []
    if RENDER:
        renderers.append(run_lib.EpisodeRenderer())

    run_lib.run_sims(env, policies, trackers, renderers, run_config)


def _get_rps_deterministic_policies(env, agent_id):
    other_agent_policies = {agent_id: {}}
    for pi_a in range(env.action_spaces[agent_id].n):
        dist = {a: 0.0 for a in range(env.action_spaces[agent_id].n)}
        dist[pi_a] = 1.0
        pi = policy_lib.FixedDistributionPolicy(
            env.model,
            ego_agent=agent_id,
            gamma=0.9,
            dist=dist,
            policy_id=f"pi_{pi_a}"
        )
        other_agent_policies[agent_id][f"pi_{pi_a}"] = pi
    return other_agent_policies


def _get_rps_biased_policies(env, agent_id, bias):
    other_agent_policies = {agent_id: {}}
    p_non_biased = (1.0 - bias) / 2
    for pi_a in range(env.action_spaces[agent_id].n):
        dist = {a: p_non_biased for a in range(env.action_spaces[agent_id].n)}
        dist[pi_a] = bias
        pi = policy_lib.FixedDistributionPolicy(
            env.model,
            ego_agent=agent_id,
            gamma=0.9,
            dist=dist,
            policy_id=f"pi_{pi_a}"
        )
        other_agent_policies[agent_id][f"pi_{pi_a}"] = pi
    return other_agent_policies


def test_bayes_accuracy_deterministic():
    """Test bayes accuracy on RPS with deterministic opponent policy.

    The accuracy should go to 100% very fast, given enough simulations are
    used.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 100

    # BAPOSGMCP has an opponent policy for each action
    other_agent_policies = _get_rps_deterministic_policies(env, 1)

    agent_0_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=0,
        gamma=0.9,
        other_policies=other_agent_policies,
        other_policy_prior=None,
        num_sims=64,
        rollout_policy=policy_lib.RandomPolicy(env.model, 1, 0.9),
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
        step_limit=rps_step_limit

    )

    # Opponent always plays first action "ROCK"
    agent_1_policy = _get_rps_deterministic_policies(env, 1)[1]["pi_0"]

    policies = [agent_0_policy, agent_1_policy]

    trackers = run_lib.get_default_trackers(policies)
    trackers.append(run_lib.BayesAccuracyTracker(2, track_per_step=True))

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=1, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, trackers, run_config)


def test_bayes_accuracy_stochastic_uniform():
    """Test bayes accuracy on RPS with stochastic opponent policy.

    If we give BAPOSGMCP the three deterministic policies for the opponent,
    then BAPOSMGPC would put all probability in one policy based on the first
    observation.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 100

    other_agent_policies = _get_rps_deterministic_policies(env, 1)

    agent_0_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=0,
        gamma=0.9,
        other_policies=other_agent_policies,
        other_policy_prior=None,
        num_sims=64,
        rollout_policy=policy_lib.RandomPolicy(env.model, 1, 0.9),
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
        step_limit=rps_step_limit

    )

    # Opponent always plays uniform random
    # Need to give this policy same ID as a policy in BAPOSGMCP other agent
    # policies so the BayesAccuracy tracker can track it properly
    agent_1_policy = policy_lib.RandomPolicy(
        env.model,
        ego_agent=1,
        gamma=0.9,
        policy_id="pi_0"   # give it same ID as a policy in BAPOSGMCP policies
    )

    policies = [agent_0_policy, agent_1_policy]

    trackers = run_lib.get_default_trackers(policies)
    trackers.append(run_lib.BayesAccuracyTracker(2, track_per_step=True))

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=1, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, trackers, run_config)


def test_bayes_accuracy_stochastic_biased():
    """Test bayes accuracy on RPS with biased stochastic opponent policy.

    Here we give BAPOSGMCP three stochastic policies for the opponent, each
    with a strong bias towards one of the three actions.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 10

    other_agent_policies = _get_rps_biased_policies(env, 1, 0.6)

    agent_0_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=0,
        gamma=0.9,
        other_policies=other_agent_policies,
        other_policy_prior=None,
        num_sims=64,
        rollout_policy=policy_lib.RandomPolicy(env.model, 1, 0.9),
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
        step_limit=rps_step_limit

    )

    # Opponent always plays biased towards "ROCK" policy
    agent_1_policy = _get_rps_biased_policies(env, 1, 0.6)[1]["pi_0"]

    policies = [agent_0_policy, agent_1_policy]

    trackers = run_lib.get_default_trackers(policies)
    trackers.append(run_lib.BayesAccuracyTracker(2, track_per_step=True))

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=10, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, trackers, run_config)


if __name__ == "__main__":
    RENDER = True
    # test_bayes_accuracy_deterministic()
    # test_bayes_accuracy_stochastic_uniform()
    test_bayes_accuracy_stochastic_biased()

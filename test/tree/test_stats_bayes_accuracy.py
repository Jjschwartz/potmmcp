import posggym

import baposgmcp.run as run_lib

import utils as test_utils


RENDER = False


def _run_sims(env, policies, num_episodes, step_limit):
    trackers = run_lib.get_default_trackers(env.n_agents, 0.9)
    trackers.append(run_lib.BayesAccuracyTracker(
        2, track_per_step=True, step_limit=step_limit
    ))
    test_utils.run_sims(
        env,
        policies,
        num_episodes=num_episodes,
        trackers=trackers,
        render=RENDER,
        **{"episode_step_limit": step_limit}
    )


def test_bayes_accuracy_deterministic():
    """Test bayes accuracy on RPS with deterministic opponent policy.

    The accuracy should go to 100% very fast, given enough simulations are
    used.
    """
    env_id = "RockPaperScissors-v0"
    env = posggym.make(env_id)
    rps_step_limit = 100

    # Opponent always plays first action "ROCK"
    agent_0_policy = test_utils.get_deterministic_policies(env, 0)["pi_0"]

    # BAPOSGMCP has an opponent policy for each action
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policy_prior=test_utils.get_deterministic_other_policy_prior(
            env, 1
        ),
        meta_policy=None,
        truncated=False,
        step_limit=rps_step_limit,
        num_sims=64
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, num_episodes=1, step_limit=rps_step_limit)


def test_bayes_accuracy_stochastic_uniform():
    """Test bayes accuracy on RPS with stochastic opponent policy.

    If we give BAPOSGMCP the three deterministic policies for the opponent,
    then BAPOSMGPC would put all probability in one policy based on the first
    observation.
    """
    env_id = "RockPaperScissors-v0"
    env = posggym.make(env_id)
    rps_step_limit = 100

    # Opponent always plays uniform random
    # Need to give this policy same ID as a policy in BAPOSGMCP other agent
    # policies so the BayesAccuracy tracker can track it properly
    agent_0_policy = test_utils.get_random_policy(env, 0)
    # give it same ID as a policy in BAPOSGMCP policies
    agent_0_policy.policy_id = "pi_0"

    # BAPOSGMCP has an opponent policy for each action
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policy_prior=test_utils.get_deterministic_other_policy_prior(
            env, 1
        ),
        meta_policy=None,
        truncated=False,
        step_limit=rps_step_limit,
        num_sims=64
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, num_episodes=1, step_limit=rps_step_limit)


def test_bayes_accuracy_stochastic_biased():
    """Test bayes accuracy on RPS with biased stochastic opponent policy.

    Here we give BAPOSGMCP three stochastic policies for the opponent, each
    with a strong bias towards one of the three actions.
    """
    env_id = "RockPaperScissors-v0"
    env = posggym.make(env_id)
    rps_step_limit = 10

    # Opponent always plays biased towards "ROCK" policy
    agent_0_policy = test_utils.get_biased_policies(env, 0, 0.3)["pi_0"]
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policy_prior=test_utils.get_biased_other_policy_prior(
            env, 1, 0.3
        ),
        meta_policy=None,
        truncated=False,
        step_limit=rps_step_limit,
        num_sims=64
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, num_episodes=1, step_limit=rps_step_limit)


if __name__ == "__main__":
    RENDER = True
    # test_bayes_accuracy_deterministic()
    # test_bayes_accuracy_stochastic_uniform()
    test_bayes_accuracy_stochastic_biased()

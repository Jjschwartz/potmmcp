import posggym

import baposgmcp.policy as P
import baposgmcp.run as run_lib

import utils as test_utils


RENDER = False


def test_bayes_accuracy_deterministic():
    """Test bayes accuracy on RPS with deterministic opponent policy.

    The accuracy should go to 100% very fast, given enough simulations are
    used.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 100

    other_policies = test_utils.get_rps_deterministic_policies(env, 1)
    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )

    # Opponent always plays first action "ROCK"
    agent_1_policy = test_utils.get_rps_deterministic_policies(
        env, 1
    )[1]["pi_0"]

    policies = [agent_0_policy, agent_1_policy]

    trackers = run_lib.get_default_trackers(policies)
    trackers.append(run_lib.BayesAccuracyTracker(2, track_per_step=True))
    run_config = run_lib.RunConfig(
        seed=0, num_episodes=1, episode_step_limit=rps_step_limit
    )

    test_utils.run_sims(env, policies, trackers, run_config, RENDER)


def test_bayes_accuracy_stochastic_uniform():
    """Test bayes accuracy on RPS with stochastic opponent policy.

    If we give BAPOSGMCP the three deterministic policies for the opponent,
    then BAPOSMGPC would put all probability in one policy based on the first
    observation.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 100

    other_policies = test_utils.get_rps_deterministic_policies(env, 1)
    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )

    # Opponent always plays uniform random
    # Need to give this policy same ID as a policy in BAPOSGMCP other agent
    # policies so the BayesAccuracy tracker can track it properly
    agent_1_policy = P.RandomPolicy(
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

    test_utils.run_sims(env, policies, trackers, run_config, RENDER)


def test_bayes_accuracy_stochastic_biased():
    """Test bayes accuracy on RPS with biased stochastic opponent policy.

    Here we give BAPOSGMCP three stochastic policies for the opponent, each
    with a strong bias towards one of the three actions.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 10

    other_policies = test_utils.get_rps_biased_policies(env, 1, 0.6)
    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )

    # Opponent always plays biased towards "ROCK" policy
    agent_1_policy = test_utils.get_rps_biased_policies(env, 1, 0.6)[1]["pi_0"]

    policies = [agent_0_policy, agent_1_policy]

    trackers = run_lib.get_default_trackers(policies)
    trackers.append(run_lib.BayesAccuracyTracker(2, track_per_step=True))
    run_config = run_lib.RunConfig(
        seed=0, num_episodes=10, episode_step_limit=rps_step_limit
    )

    test_utils.run_sims(env, policies, trackers, run_config, RENDER)


if __name__ == "__main__":
    RENDER = True
    # test_bayes_accuracy_deterministic()
    # test_bayes_accuracy_stochastic_uniform()
    test_bayes_accuracy_stochastic_biased()

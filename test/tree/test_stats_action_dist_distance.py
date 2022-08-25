import posggym

import baposgmcp.policy as P
import baposgmcp.run as run_lib

import utils as test_utils

RENDER = False


def _run_sims(env, policies, run_config):
    trackers = run_lib.get_default_trackers(policies)
    trackers.append(
        run_lib.ActionDistributionDistanceTracker(
            env.n_agents,
            track_per_step=True,
            step_limit=run_config.episode_step_limit
        )
    )
    test_utils.run_sims(env, policies, trackers, run_config, RENDER)


def test_action_dist_distance_single_policy():
    """Test action distribution accuracy on RPS.

    In this case the BAPOSGMCP other agent policy matches exactly the true
    other agent policy so the distance should be 0.0 for all steps.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 10

    other_policies = test_utils.get_rps_random_policy(env, 1)
    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )

    agent_1_policy = P.RandomPolicy(
        env.model, 1, 0.9, policy_id="pi_-1"
    )

    policies = [agent_0_policy, agent_1_policy]

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, run_config)


def test_action_dist_distance_rps_deterministic():
    """Test action dist distance on RPS with deterministic opponent policy.

    The distance should go to 0.0 very fast (actually after a single
    observation).
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 20

    # BAPOSGMCP has an opponent policy for each action
    other_policies = test_utils.get_rps_deterministic_policies(env, 1)

    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )

    # Opponent always plays first action "ROCK"
    agent_1_policy = test_utils.get_rps_deterministic_policies(
        env, 1
    )[1]["pi_0"]

    policies = [agent_0_policy, agent_1_policy]

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, run_config)


def test_action_dist_distance_rps_stochastic_biased():
    """Test action distance on RPS with biased stochastic opponent policy.

    Here we give BAPOSGMCP three stochastic policies for the opponent, each
    with a strong bias towards one of the three actions.

    Since the true opponent policy is one of the policies in the belief of the
    BAPOSGMCP agent, the distance should drop to 0.0, but it will take more
    steps.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 100

    other_policies = test_utils.get_rps_biased_policies(env, 1, 0.6)

    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )

    # Opponent always plays biased towards "ROCK" policy
    agent_1_policy = test_utils.get_rps_biased_policies(env, 1, 0.6)[1]["pi_0"]

    policies = [agent_0_policy, agent_1_policy]

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, run_config)


def test_action_dist_distance_rps_stochastic_biased2():
    """Test same as above except BAPOSGMCP doesnt have true policy in belief.

    Here we give BAPOSGMCP three stochastic policies for the opponent, each
    with a strong bias towards one of the three actions.

    The true policy is also biased but less so. So here we expect the distance
    shouldn't drop to 0.0 or rather it probably will due to sampling error,
    but it should happen slower or not at all.

    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 10

    other_policies = test_utils.get_rps_biased_policies(env, 1, 0.6)

    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )

    # Opponent always plays biased towards "ROCK" policy
    # Bias is different to bias in other_agent_policies
    agent_1_policy = test_utils.get_rps_biased_policies(
        env, 1, 0.45
    )[1]["pi_0"]

    policies = [agent_0_policy, agent_1_policy]

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, run_config)


if __name__ == "__main__":
    RENDER = True
    # test_action_dist_distance_single_policy()
    # test_action_dist_distance_rps_deterministic()
    test_action_dist_distance_rps_stochastic_biased()
    # test_action_dist_distance_rps_stochastic_biased2()
    # test_history_accuracy_small()

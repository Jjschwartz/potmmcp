import posggym

import baposgmcp.run as run_lib

import utils as test_utils

RENDER = False


def _run_sims(env, policies, num_episodes, step_limit):
    trackers = run_lib.get_default_trackers(env.n_agents, 0.9)
    trackers.append(
        run_lib.ActionDistributionDistanceTracker(
            env.n_agents,
            track_per_step=True,
            step_limit=step_limit
        )
    )
    test_utils.run_sims(
        env,
        policies,
        num_episodes=num_episodes,
        trackers=trackers,
        render=RENDER,
        **{"episode_step_limit": step_limit}
    )


def test_action_dist_distance_single_policy():
    """Test action distribution accuracy on RPS.

    In this case the BAPOSGMCP other agent policy matches exactly the true
    other agent policy so the distance should be 0.0 for all steps.
    """
    env_id = "RockPaperScissors-v0"
    env = posggym.make(env_id)
    rps_step_limit = 10

    agent_0_policy = test_utils.get_random_policy(env, 0)
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policy_prior=None,
        meta_policy=None,
        truncated=False,
        step_limit=rps_step_limit,
        num_sims=64
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, num_episodes=5, step_limit=rps_step_limit)


def test_action_dist_distance_rps_deterministic():
    """Test action dist distance on RPS with deterministic opponent policy.

    The distance should go to 0.0 very fast (actually after a single
    observation).
    """
    env_id = "RockPaperScissors-v0"
    env = posggym.make(env_id)
    rps_step_limit = 20

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
    _run_sims(env, policies, num_episodes=5, step_limit=rps_step_limit)


def test_action_dist_distance_rps_stochastic_biased():
    """Test action distance on RPS with biased stochastic opponent policy.

    Here we give BAPOSGMCP three stochastic policies for the opponent, each
    with a strong bias towards one of the three actions.

    Since the true opponent policy is one of the policies in the belief of the
    BAPOSGMCP agent, the distance should drop to 0.0, but it will take more
    steps.
    """
    env_id = "RockPaperScissors-v0"
    env = posggym.make(env_id)
    rps_step_limit = 100

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


def test_action_dist_distance_rps_stochastic_biased2():
    """Test same as above except BAPOSGMCP doesnt have true policy in belief.

    Here we give BAPOSGMCP three stochastic policies for the opponent, each
    with a strong bias towards one of the three actions.

    The true policy is also biased but less so. So here we expect the distance
    shouldn't drop to 0.0 or rather it probably will due to sampling error,
    but it should happen slower or not at all.

    """
    env_id = "RockPaperScissors-v0"
    env = posggym.make(env_id)
    rps_step_limit = 10

    # Opponent always plays biased towards "ROCK" policy
    # Bias is different to bias in other_agent_policies
    agent_0_policy = test_utils.get_biased_policies(env, 0, 0.15)["pi_0"]
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
    _run_sims(env, policies, num_episodes=5, step_limit=rps_step_limit)


if __name__ == "__main__":
    RENDER = True
    # test_action_dist_distance_single_policy()
    # test_action_dist_distance_rps_deterministic()
    test_action_dist_distance_rps_stochastic_biased()
    # test_action_dist_distance_rps_stochastic_biased2()
    # test_history_accuracy_small()

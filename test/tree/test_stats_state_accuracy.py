import posggym

import baposgmcp.run as run_lib

import utils as test_utils

RENDER = False


def _run_sims(env, policies, num_episodes, step_limit):
    trackers = run_lib.get_default_trackers(env.n_agents, 0.9)
    trackers.append(
        run_lib.BeliefStateAccuracyTracker(
            env.n_agents, track_per_step=True, step_limit=step_limit
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


def test_state_accuracy_single_state():
    """Test state accuracy on RPS which has only a single state.

    The accuracy should be 100%
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


def test_state_accuracy_small():
    """Test state accuracy on small environment."""
    env_id = "TwoPaths4x4-v1"
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


if __name__ == "__main__":
    RENDER = True
    test_state_accuracy_single_state()
    test_state_accuracy_small()

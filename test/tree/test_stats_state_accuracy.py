import posggym

import potmmcp.run as run_lib

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


def test_state_accuracy():
    """Test state accuracy,"""
    env_id = "PursuitEvasion16x16-v0"
    env = posggym.make(env_id, max_episode_steps=10)

    agent_0_policy = test_utils.get_random_policy(env, 0)
    agent_1_policy = test_utils.get_random_potmmcp(
        env,
        1,
        other_policy_prior=None,
        meta_policy=None,
        truncated=False,
        step_limit=10,
        num_sims=64,
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, num_episodes=5, step_limit=10)


if __name__ == "__main__":
    RENDER = True
    test_state_accuracy()

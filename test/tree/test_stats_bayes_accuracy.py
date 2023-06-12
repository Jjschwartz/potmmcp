import posggym

import potmmcp.run as run_lib

import utils as test_utils


RENDER = False


def _run_sims(env, policies, num_episodes, step_limit):
    trackers = run_lib.get_default_trackers(env.n_agents, 0.9)
    trackers.append(
        run_lib.BayesAccuracyTracker(2, track_per_step=True, step_limit=step_limit)
    )
    test_utils.run_sims(
        env,
        policies,
        num_episodes=num_episodes,
        trackers=trackers,
        render=RENDER,
        **{"episode_step_limit": step_limit}
    )


def test_bayes_accuracy():
    """Test bayes accuracy."""
    env_id = "PursuitEvasion16x16-v0"
    env = posggym.make(env_id, max_episode_steps=10)

    # Opponent always plays first action "ROCK"
    agent_0_policy = test_utils.get_deterministic_policies(env, 0)["pi_0"]

    # POTMMCP has an opponent policy for each action
    agent_1_policy = test_utils.get_random_potmmcp(
        env,
        1,
        other_policy_prior=test_utils.get_deterministic_other_policy_prior(env, 1),
        meta_policy=None,
        truncated=False,
        step_limit=10,
        num_sims=64,
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, num_episodes=1, step_limit=10)


if __name__ == "__main__":
    RENDER = True
    test_bayes_accuracy()

import logging

import posggym

import baposgmcp.run as run_lib
import baposgmcp.tree as tree_lib
import baposgmcp.policy as policy_lib

RENDER = False


def _run_sims(env, policies, run_config):
    logging.basicConfig(level=logging.INFO-1, format='%(message)s')

    trackers = run_lib.get_default_trackers(policies)
    trackers.append(
        run_lib.BeliefHistoryAccuracyTracker(
            env.n_agents,
            track_per_step=True,
            step_limit=run_config.episode_step_limit
        )
    )

    renderers = []
    if RENDER:
        renderers.append(run_lib.EpisodeRenderer())

    run_lib.run_sims(env, policies, trackers, renderers, run_config)


def test_history_accuracy_fully_obs():
    """Test history accuracy on RPS.

    In RPS agents observe each others actions and so the history is fully
    observed, hence the accuracy should be 100%.
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 10

    agent_0_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=0,
        gamma=0.9,
        other_policies={
            1: {"pi_-1": policy_lib.RandomPolicy(env.model, 1, 0.9)}
        },
        other_policy_prior=None,
        num_sims=64,
        rollout_policy=policy_lib.RandomPolicy(env.model, 1, 0.9),
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
        step_limit=rps_step_limit

    )

    agent_1_policy = policy_lib.RandomPolicy(
        env.model, 1, 0.9, policy_id="pi_-1"
    )

    policies = [agent_0_policy, agent_1_policy]

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, run_config)


def test_history_accuracy_small():
    """Test history accuracy on small environment.

    Here we'd expect the accuracy to decrease over time.
    """
    env_name = "TwoPaths4x4-v0"
    env = posggym.make(env_name)
    rps_step_limit = 20

    agent_0_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=0,
        gamma=0.9,
        other_policies={
            1: {"pi_-1": policy_lib.RandomPolicy(env.model, 1, 0.9)}
        },
        other_policy_prior=None,
        num_sims=64,
        rollout_policy=policy_lib.RandomPolicy(env.model, 1, 0.9),
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
        step_limit=rps_step_limit

    )

    agent_1_policy = policy_lib.RandomPolicy(
        env.model, 1, 0.9, policy_id="pi_-1"
    )

    policies = [agent_0_policy, agent_1_policy]

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, run_config)


if __name__ == "__main__":
    RENDER = True
    # test_history_accuracy_fully_obs()
    test_history_accuracy_small()

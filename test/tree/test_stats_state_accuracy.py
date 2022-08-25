import posggym

import baposgmcp.policy as P
import baposgmcp.run as run_lib

import utils as test_utils

RENDER = False


def _run_sims(env, policies, run_config):
    trackers = run_lib.get_default_trackers(policies)
    trackers.append(
        run_lib.BeliefStateAccuracyTracker(env.n_agents, track_per_step=True)
    )
    test_utils.run_sims(env, policies, trackers, run_config, RENDER)


def test_state_accuracy_single_state():
    """Test state accuracy on RPS which has only a single state.

    The accuracy should be 100%
    """
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 10

    other_policies = test_utils.get_rps_random_policy(env, 1)
    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )
    agent_1_policy = P.RandomPolicy(env.model, 1, 0.9, policy_id="pi_-1")

    policies = [agent_0_policy, agent_1_policy]
    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )
    _run_sims(env, policies, run_config)


def test_state_accuracy_small():
    """Test state accuracy on small environment."""
    env_name = "TwoPaths4x4-v1"
    env = posggym.make(env_name)
    rps_step_limit = 10

    other_policies = test_utils.get_rps_random_policy(env, 1)
    agent_0_policy = test_utils.get_random_baposgmcp(
        env, 0, other_policies, False, rps_step_limit
    )
    agent_1_policy = P.RandomPolicy(env.model, 1, 0.9, policy_id="pi_-1")

    policies = [agent_0_policy, agent_1_policy]

    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )

    _run_sims(env, policies, run_config)


if __name__ == "__main__":
    RENDER = True
    test_state_accuracy_single_state()
    test_state_accuracy_small()

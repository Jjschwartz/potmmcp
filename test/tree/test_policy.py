from collections import defaultdict

import posggym

import potmmcp.run as run_lib
from potmmcp.meta_policy import DictMetaPolicy

import utils as test_utils

RENDER = False


def _run_sims(env, policies, num_episodes=10, step_limit=None):
    test_utils.run_sims(
        env,
        policies,
        num_episodes=num_episodes,
        trackers=run_lib.get_default_trackers(env.n_agents, 0.9),
        render=RENDER,
        **{"episode_step_limit": step_limit}
    )


def test_with_single_random_policy():
    """Test POTMMCP tree with only a single random other agent policy."""
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
        num_sims=128,
    )
    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, step_limit=10)


def test_with_single_random_policy_truncated():
    """Test POTMMCP tree with only a single random other agent policy."""
    env_id = "PursuitEvasion16x16-v0"
    env = posggym.make(env_id, max_episode_steps=10)

    agent_0_policy = test_utils.get_random_policy(env, 0)
    agent_1_policy = test_utils.get_random_potmmcp(
        env,
        1,
        other_policy_prior=None,
        meta_policy=None,
        truncated=True,
        step_limit=10,
        num_sims=128,
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, step_limit=10)


def test_with_multiple_random_policies():
    """Test POTMMCP tree with multiple random other agent policies."""
    env_id = "PursuitEvasion16x16-v0"
    env = posggym.make(env_id, max_episode_steps=10)

    agent_0_policy = test_utils.get_random_policy(env, 0)

    other_policy_prior = test_utils.get_biased_other_policy_prior(env, 1, 0.2)
    agent_1_policy = test_utils.get_random_potmmcp(
        env,
        1,
        other_policy_prior=other_policy_prior,
        meta_policy=None,
        truncated=False,
        step_limit=10,
        num_sims=128,
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, step_limit=10)


def test_with_multiple_random_opponent_and_rollout_policies():
    """Test POTMMCP tree with multiple random other agent policies."""
    env_id = "PursuitEvasion16x16-v0"
    env = posggym.make(env_id, max_episode_steps=10)

    agent_0_policy = test_utils.get_random_policy(env, 0)

    other_policy_prior = test_utils.get_biased_other_policy_prior(env, 1, 0.2)
    ego_policies = test_utils.get_biased_policies(env, 1, 0.2)

    meta_default = {pi_id: 1.0 / len(ego_policies) for pi_id in ego_policies}
    meta_policy = DictMetaPolicy(
        env.model, 1, ego_policies, defaultdict(lambda: meta_default)
    )
    agent_1_policy = test_utils.get_random_potmmcp(
        env,
        1,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        truncated=False,
        step_limit=10,
        num_sims=128,
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, step_limit=10)


def test_with_three_other_agents():
    """Test POTMMCP tree with multiple random other agent policies."""
    env_id = "PredatorPrey10x10-P4-p3-s3-coop-v0"
    env = posggym.make(env_id, max_episode_steps=10)

    agent_0_policy = test_utils.get_random_policy(env, 0)
    agent_1_policy = test_utils.get_random_policy(env, 1)
    agent_2_policy = test_utils.get_random_potmmcp(
        env,
        2,
        other_policy_prior=None,
        meta_policy=None,
        truncated=False,
        step_limit=10,
        num_sims=128,
    )
    agent_3_policy = test_utils.get_random_policy(env, 3)

    policies = [agent_0_policy, agent_1_policy, agent_2_policy, agent_3_policy]
    _run_sims(env, policies, step_limit=10)


if __name__ == "__main__":
    RENDER = True
    # test_with_single_random_policy()
    # test_with_single_random_policy_truncated()
    # test_with_multiple_random_policies()
    # test_with_multiple_random_opponent_and_rollout_policies()
    test_with_three_other_agents()

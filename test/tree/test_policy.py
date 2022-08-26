from collections import defaultdict

import posggym

import baposgmcp.run as run_lib
from baposgmcp.meta_policy import DictMetaPolicy

import utils as test_utils

RENDER = False


def _run_sims(env, policies, run_config=None):
    trackers = run_lib.get_default_trackers(policies)

    if run_config is None:
        run_config = run_lib.RunConfig(seed=0, num_episodes=10)

    test_utils.run_sims(env, policies, trackers, run_config, RENDER)


def test_with_single_random_policy():
    """Test BAPOSGMCP tree with only a single random other agent policy."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = test_utils.get_random_policy(env, 0)
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policies=None,
        meta_policy=None,
        truncated=False,
        step_limit=None,
        num_sims=128
    )
    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


def test_action_first_with_single_random_policy():
    """Test BAPOSGMCP in action first env with a random other agent policy."""
    env_name = "RockPaperScissors-v0"
    env = posggym.make(env_name)
    rps_step_limit = 10
    run_config = run_lib.RunConfig(
        seed=0, num_episodes=5, episode_step_limit=rps_step_limit
    )

    agent_0_policy = test_utils.get_random_policy(env, 0)
    agent_1_policy = test_utils.get_random_baposgmcp(
        env, 1,
        other_policies=None,
        meta_policy=None,
        truncated=False,
        step_limit=rps_step_limit,
        num_sims=64
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies, run_config)


def test_with_single_random_policy_truncated():
    """Test BAPOSGMCP tree with only a single random other agent policy."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = test_utils.get_random_policy(env, 0)
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policies=None,
        meta_policy=None,
        truncated=True,
        step_limit=None,
        num_sims=128
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


def test_with_multiple_random_policies():
    """Test BAPOSGMCP tree with multiple random other agent policies."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = test_utils.get_random_policy(env, 0)

    other_policies = test_utils.get_biased_other_policies(env, 1, 0.2)
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policies=other_policies,
        meta_policy=None,
        truncated=False,
        step_limit=None,
        num_sims=128
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


def test_with_multiple_random_opponent_and_rollout_policies():
    """Test BAPOSGMCP tree with multiple random other agent policies."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = test_utils.get_random_policy(env, 0)

    other_policies = test_utils.get_biased_other_policies(env, 1, 0.2)
    ego_policies = test_utils.get_biased_policies(env, 1, 0.2)

    meta_default = {pi_id: 1.0 / len(ego_policies) for pi_id in ego_policies}
    meta_policy = DictMetaPolicy(
        env.model, 1, ego_policies, defaultdict(lambda: meta_default)
    )
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policies=other_policies,
        meta_policy=meta_policy,
        truncated=False,
        step_limit=None,
        num_sims=128
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


def test_with_three_other_agents():
    """Test BAPOSGMCP tree with multiple random other agent policies."""
    env_name = "Driving4x4Intersection-n3-v0"
    env = posggym.make(env_name)

    agent_0_policy = test_utils.get_random_policy(env, 0)
    agent_1_policy = test_utils.get_random_policy(env, 1)
    agent_2_policy = test_utils.get_random_baposgmcp(
        env,
        2,
        other_policies=None,
        meta_policy=None,
        truncated=False,
        step_limit=None,
        num_sims=128
    )

    policies = [agent_0_policy, agent_1_policy, agent_2_policy]
    _run_sims(env, policies)


if __name__ == "__main__":
    RENDER = True
    # test_with_single_random_policy()
    # test_with_single_random_policy_truncated()
    # test_with_multiple_random_policies()
    # test_with_multiple_random_opponent_and_rollout_policies()
    test_with_three_other_agents()

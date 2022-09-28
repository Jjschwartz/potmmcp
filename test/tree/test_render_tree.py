import logging
from collections import defaultdict

import posggym

import baposgmcp.run as run_lib
from baposgmcp.meta_policy import DictMetaPolicy

import utils as test_utils

SEARCH_TREE_DEPTH = 3
RENDER = False


def _run_sims(env, policies):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    trackers = run_lib.get_default_trackers(env.n_agents, 0.9)

    renderers = []
    if RENDER:
        renderers = [
            run_lib.EpisodeRenderer(),
            run_lib.SearchTreeRenderer(SEARCH_TREE_DEPTH),
            run_lib.PauseRenderer()
        ]

    run_lib.run_episodes(
        env,
        policies,
        num_episodes=10,
        trackers=trackers,
        renderers=renderers
    )


def test_with_multiple_random_opponent_and_rollout_policies():
    """Test BAPOSGMCP tree with multiple random other agent policies."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = test_utils.get_random_policy(env, 0)

    other_policy_prior = test_utils.get_biased_other_policy_prior(env, 1, 0.2)
    ego_policies = test_utils.get_biased_policies(env, 1, 0.2)

    meta_default = {pi_id: 1.0 / len(ego_policies) for pi_id in ego_policies}
    meta_policy = DictMetaPolicy(
        env.model, 1, ego_policies, defaultdict(lambda: meta_default)
    )
    agent_1_policy = test_utils.get_random_baposgmcp(
        env,
        1,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        truncated=False,
        step_limit=None,
        num_sims=128
    )

    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


if __name__ == "__main__":
    RENDER = True
    test_with_multiple_random_opponent_and_rollout_policies()

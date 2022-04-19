import logging

import posggym

from baposgmcp import runner
import baposgmcp.tree as tree_lib
import baposgmcp.stats as stats_lib
import baposgmcp.policy as policy_lib
import baposgmcp.render as render_lib

RENDER = False


def _run_sims(env, policies):
    logging.basicConfig(level="INFO", format='%(message)s')
    trackers = stats_lib.get_default_trackers(policies)

    renderers = []
    if RENDER:
        renderers.append(render_lib.EpisodeRenderer())

    runner.run_sims(
        env,
        policies,
        trackers,
        renderers,
        run_config=runner.RunConfig(seed=0, num_episodes=10)
    )


def test_with_single_random_policy():
    """Test BAPOSGMCP tree with only a single random other agent policy."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = policy_lib.RandomPolicy(
        env.model,
        ego_agent=0,
        gamma=0.9
    )
    agent_1_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=1,
        gamma=0.9,
        other_policies={
            0: {'pi_-1_0': policy_lib.RandomPolicy(env.model, 0, 0.9)}
        },
        other_policy_prior=None,
        num_sims=64,
        rollout_policy=policy_lib.RandomPolicy(env.model, 0, 0.9),
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
    )
    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


def test_with_single_random_policy_truncated():
    """Test BAPOSGMCP tree with only a single random other agent policy."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = policy_lib.RandomPolicy(
        env.model,
        ego_agent=0,
        gamma=0.9
    )
    agent_1_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=1,
        gamma=0.9,
        other_policies={
            0: {'pi_-1_0': policy_lib.RandomPolicy(env.model, 0, 0.9)}
        },
        other_policy_prior=None,
        num_sims=64,
        rollout_policy=policy_lib.RandomPolicy(env.model, 0, 0.9),
        c_init=1.0,
        c_base=100.0,
        truncated=True,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
    )
    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


def test_with_multiple_random_policies():
    """Test BAPOSGMCP tree with multiple random other agent policies."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = policy_lib.RandomPolicy(
        env.model,
        ego_agent=0,
        gamma=0.9
    )

    agent_0_policies = {'pi_-1_0': policy_lib.RandomPolicy(env.model, 0, 0.9)}
    action_space = list(range(env.action_spaces[0].n))
    n_actions = len(action_space)
    for biased_a in [2, 3]:    # for actions RIGHT, LEFT
        a_dist = {}
        for a in action_space:
            if a == biased_a:
                pr_a = (1.0 / n_actions) + 0.2
            else:
                pr_a = (1.0 / n_actions) - (0.2 / (n_actions-1))
            a_dist[a] = pr_a
        pi = policy_lib.FixedDistributionPolicy(env.model, 0, 0.9, a_dist)
        agent_0_policies[f"pi_{biased_a}_0"] = pi

    agent_1_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=1,
        gamma=0.9,
        other_policies={0: agent_0_policies},
        other_policy_prior=None,
        num_sims=64,
        rollout_policy=policy_lib.RandomPolicy(env.model, 0, 0.9),
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
    )
    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


if __name__ == "__main__":
    RENDER = True
    test_with_single_random_policy()
    test_with_single_random_policy_truncated()
    test_with_multiple_random_policies()

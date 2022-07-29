import logging

import posggym

import baposgmcp.run as run_lib
import baposgmcp.tree as tree_lib
import baposgmcp.policy as policy_lib

RENDER = False


def _run_sims(env, policies):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    trackers = run_lib.get_default_trackers(policies)

    renderers = []
    if RENDER:
        renderers.append(run_lib.EpisodeRenderer())

    run_lib.run_sims(
        env,
        policies,
        trackers,
        renderers,
        run_config=run_lib.RunConfig(seed=0, num_episodes=10)
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
        num_sims=128,
        other_policies={
            0: {'pi_-1': policy_lib.RandomPolicy(env.model, 0, 0.9)}
        },
        other_policy_prior=None,
        rollout_policies={
            'pi_-1': policy_lib.RandomPolicy(env.model, 0, 0.9),
        },
        rollout_selection={'pi_-1': 'pi_-1'},
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
        num_sims=128,
        other_policies={
            0: {'pi_-1': policy_lib.RandomPolicy(env.model, 0, 0.9)}
        },
        other_policy_prior=None,
        rollout_policies={
            'pi_-1': policy_lib.RandomPolicy(env.model, 0, 0.9),
        },
        rollout_selection={'pi_-1': 'pi_-1'},
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

    agent_0_policies = {'pi_-1': policy_lib.RandomPolicy(env.model, 0, 0.9)}
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
        agent_0_policies[f"pi_{biased_a}"] = pi

    agent_1_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=1,
        gamma=0.9,
        num_sims=128,
        other_policies={0: agent_0_policies},
        other_policy_prior=None,
        rollout_policies={
            'pi_-1': policy_lib.RandomPolicy(env.model, 0, 0.9),
        },
        rollout_selection={
            'pi_-1': 'pi_-1', 'pi_2': 'pi_-1', 'pi_3': 'pi_-1'
        },
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
    )
    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


def test_with_multiple_random_opponent_and_rollout_policies():
    """Test BAPOSGMCP tree with multiple random other agent policies."""
    env_name = "TwoPaths3x3-v0"
    env = posggym.make(env_name)

    agent_0_policy = policy_lib.RandomPolicy(
        env.model,
        ego_agent=0,
        gamma=0.9
    )

    agent_0_policies = {'pi_-1': policy_lib.RandomPolicy(env.model, 0, 0.9)}
    rollout_policies = {'pi_-1': policy_lib.RandomPolicy(env.model, 0, 0.9)}
    rollout_selection = {'pi_-1': 'pi_-1'}

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
        agent_0_policies[f"pi_{biased_a}"] = pi

        rollout_pi = policy_lib.FixedDistributionPolicy(
            env.model, 0, 0.9, a_dist
        )
        rollout_policies[f"pi_{biased_a}"] = rollout_pi
        rollout_selection[f"pi_{biased_a}"] = f"pi_{biased_a}"

    agent_1_policy = tree_lib.BAPOSGMCP(
        env.model,
        ego_agent=1,
        gamma=0.9,
        num_sims=128,
        other_policies={0: agent_0_policies},
        other_policy_prior=None,
        rollout_policies=rollout_policies,
        rollout_selection=rollout_selection,
        c_init=1.0,
        c_base=100.0,
        truncated=False,
        reinvigorator=tree_lib.BABeliefRejectionSampler(env.model),
    )
    policies = [agent_0_policy, agent_1_policy]
    _run_sims(env, policies)


if __name__ == "__main__":
    RENDER = True
    # test_with_single_random_policy()
    # test_with_single_random_policy_truncated()
    # test_with_multiple_random_policies()
    test_with_multiple_random_opponent_and_rollout_policies()

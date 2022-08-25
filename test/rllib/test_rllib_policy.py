"""Test RLLib Policy.

Specifically, tests:
- creating a baposgmcp.rllib.RllibPolicy from and rllib.Trainer
- runnning the policy using the baposgmcp runner functions
- running RllibPolicy with BAPOSGMCP Tree class
"""
import logging

import ray

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy

import posggym
from posggym.wrappers import FlattenObservation
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

import baposgmcp.run as run_lib
import baposgmcp.tree as tree_lib
import baposgmcp.rllib as ba_rllib
import baposgmcp.policy as P


RENDER = False
TEST_ENV_NAME = "TwoPaths3x3-v0"
# Sanity check
posggym.make(TEST_ENV_NAME)


def _env_creator(config):
    env = posggym.make(TEST_ENV_NAME)
    env = FlattenObservation(env)
    return RllibMultiAgentEnv(env)


def _get_env():
    return posggym.make(TEST_ENV_NAME)


ray.init()
register_env(TEST_ENV_NAME, _env_creator)


def _get_ppo_trainer():
    env_name = TEST_ENV_NAME

    env_config = {"env_name": env_name}
    sample_env = _env_creator(env_config)
    # obs and action spaces are the same for both agent in TwoPaths env
    obs_space = sample_env.observation_space["0"]
    act_space = sample_env.action_space["0"]

    agent_0_policy_spec = PolicySpec(RandomPolicy, obs_space, act_space, {})
    agent_1_policy_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})

    def _policy_mapping_fn(agent_id, episode, worker, **kwargs):
        for policy_id in episode.policy_map.keys():
            if policy_id.endswith(agent_id):
                return policy_id
        raise AssertionError

    trainer = PPOTrainer(
        env=env_name,
        config={
            "env_config": env_config,
            "gamma": 0.9,
            "num_workers": 0,
            "num_envs_per_worker": 4,
            "rollout_fragment_length": 10,
            "train_batch_size": 200,
            "metrics_num_episodes_for_smoothing": 200,
            "multiagent": {
                "policies": {
                    "pi_0": agent_0_policy_spec,
                    "pi_1": agent_1_policy_spec
                },
                "policy_mapping_fn": _policy_mapping_fn,
                "policies_to_train": ["pi_1"],
            },
            "model": {
                "use_lstm": True,
                "vf_share_layers": True,
            },
            "num_sgd_iter": 6,
            "vf_loss_coeff": 0.01,
            "observation_filter": "MeanStdFilter",
            "num_gpus": 0,
            "log_level": "WARN",
            "framework": "torch",
        }
    )

    for iteration in range(1):
        print(f"== Iteration {iteration} ==")
        result = trainer.train()
        print(pretty_print(result))

    return trainer


def _get_baposgmcp(model,
                   agent_id,
                   other_policies,
                   rollout_policies,
                   rollout_selection,
                   truncated):
    return tree_lib.BAPOSGMCP(
        model=model,
        ego_agent=agent_id,
        gamma=0.9,
        num_sims=8,
        other_policies=other_policies,
        other_policy_prior=None,
        rollout_policies=rollout_policies,
        rollout_selection=rollout_selection,
        c_init=1.0,
        c_base=100.0,
        truncated=truncated,
        reinvigorator=tree_lib.BABeliefRejectionSampler(model),
    )


def _run_sims(env, policies):
    logging.basicConfig(level=logging.INFO-10, format='%(message)s')
    trackers = run_lib.get_default_trackers(policies)

    renderers = []
    if RENDER:
        renderers.append(run_lib.EpisodeRenderer())

    run_lib.run_sims(
        env,
        policies,
        trackers,
        renderers,
        run_config=run_lib.RunConfig(seed=0, num_episodes=3)
    )


def test_rllib_ppopolicy():
    """Run integration test for PPORLlibPolicy."""
    sample_env = _get_env()
    trainer = _get_ppo_trainer()

    env_model = sample_env.model
    agent_0_policy = P.RandomPolicy(
        env_model,
        ego_agent=0,
        gamma=0.9,
        policy_id='pi_-1_0',
    )
    agent_1_policy = ba_rllib.PPORllibPolicy(
        env_model,
        ego_agent=1,
        gamma=0.9,
        policy_id='pi_0_1',
        policy=trainer.get_policy("pi_1"),
        preprocessor=ba_rllib.get_flatten_preprocessor(
            env_model.observation_spaces[1]
        )
    )
    policies = [agent_0_policy, agent_1_policy]
    _run_sims(sample_env, policies)

    joint_obs = sample_env.reset()
    agent_1_policy.reset()
    agent_1_policy.step(joint_obs[1])
    agent_1_policy.get_pi(None)
    agent_1_policy.get_value(None)


def test_rllib_ppopolicy_with_tree():
    """Run integration test for PPORLlibPolicy."""
    sample_env = _get_env()
    trainer = _get_ppo_trainer()

    env_model = sample_env.model
    agent_1_random_policy = P.RandomPolicy(
        env_model,
        ego_agent=1,
        gamma=0.9,
        policy_id='pi_-1_1'
    )
    agent_1_rllib_policy = ba_rllib.PPORllibPolicy(
        env_model,
        ego_agent=1,
        gamma=0.9,
        policy_id='pi_0_1',
        policy=trainer.get_policy("pi_1"),
        preprocessor=ba_rllib.get_flatten_preprocessor(
            env_model.observation_spaces[1]
        )
    )

    agent_0_policy = _get_baposgmcp(
        env_model,
        0,
        other_policies={1: {'pi_0_1': agent_1_rllib_policy}},
        rollout_policies={
            'pi_-1_0': P.RandomPolicy(env_model, 0, 0.9, policy_id='pi_-1_0')
        },
        rollout_selection={'pi_0_1': 'pi_-1_0'},
        truncated=False,
    )

    policies = [agent_0_policy, agent_1_random_policy]
    _run_sims(sample_env, policies)


def test_rllib_ppopolicy_with_tree_truncated():
    """Run integration test for PPORLlibPolicy."""
    sample_env = _get_env()
    trainer = _get_ppo_trainer()

    env_model = sample_env.model
    agent_1_random_policy = P.RandomPolicy(
        env_model,
        ego_agent=1,
        gamma=0.9,
        policy_id='pi_-1_1'
    )
    agent_1_rllib_policy = ba_rllib.PPORllibPolicy(
        env_model,
        ego_agent=1,
        gamma=0.9,
        policy_id='pi_0_1',
        policy=trainer.get_policy("pi_1"),
        preprocessor=ba_rllib.get_flatten_preprocessor(
            env_model.observation_spaces[1]
        )
    )

    agent_0_policy = _get_baposgmcp(
        env_model,
        0,
        other_policies={1: {'pi_0_1': agent_1_rllib_policy}},
        rollout_policies={
            'pi_-1_0': P.RandomPolicy(env_model, 0, 0.9, policy_id='pi_-1_0')
        },
        rollout_selection={'pi_0_1': 'pi_-1_0'},
        truncated=True,
    )

    policies = [agent_0_policy, agent_1_random_policy]
    _run_sims(sample_env, policies)


def test_rllib_ppopolicy_with_tree_as_rollout():
    """Run integration test for BAPOSGMCP using PPO Policy as rollout pi."""
    sample_env = _get_env()
    trainer = _get_ppo_trainer()

    env_model = sample_env.model
    agent_0_random_policy = P.RandomPolicy(
        env_model,
        ego_agent=0,
        gamma=0.9,
        policy_id='pi_-1_0'
    )
    agent_1_rllib_policy = ba_rllib.PPORllibPolicy(
        env_model,
        ego_agent=1,
        gamma=0.9,
        policy_id='pi_0_1',
        policy=trainer.get_policy("pi_1"),
        preprocessor=ba_rllib.get_flatten_preprocessor(
            env_model.observation_spaces[1]
        )
    )

    agent_1_policy = _get_baposgmcp(
        env_model,
        0,
        other_policies={1: {'pi_-1_0': agent_0_random_policy}},
        rollout_policies={'pi_0_1': agent_1_rllib_policy},
        rollout_selection={'pi_-1_0': 'pi_0_1'},
        truncated=False
    )

    policies = [agent_0_random_policy, agent_1_policy]
    _run_sims(sample_env, policies)


def test_rllib_ppopolicy_with_tree_as_rollout_truncated():
    """Run integration test for BAPOSGMCP using PPO Policy as rollout pi."""
    sample_env = _get_env()
    trainer = _get_ppo_trainer()

    env_model = sample_env.model
    agent_0_random_policy = P.RandomPolicy(
        env_model,
        ego_agent=0,
        gamma=0.9,
        policy_id='pi_-1_0'
    )
    agent_1_rllib_policy = ba_rllib.PPORllibPolicy(
        env_model,
        ego_agent=1,
        gamma=0.9,
        policy_id='pi_0_1',
        policy=trainer.get_policy("pi_1"),
        preprocessor=ba_rllib.get_flatten_preprocessor(
            env_model.observation_spaces[1]
        )
    )

    agent_1_policy = _get_baposgmcp(
        env_model,
        0,
        other_policies={1: {'pi_-1_0': agent_0_random_policy}},
        rollout_policies={'pi_0_1': agent_1_rllib_policy},
        rollout_selection={'pi_-1_0': 'pi_0_1'},
        truncated=True
    )

    policies = [agent_0_random_policy, agent_1_policy]
    _run_sims(sample_env, policies)


if __name__ == "__main__":
    RENDER = True
    test_rllib_ppopolicy()
    test_rllib_ppopolicy_with_tree()
    test_rllib_ppopolicy_with_tree_truncated()
    test_rllib_ppopolicy_with_tree_as_rollout()
    test_rllib_ppopolicy_with_tree_as_rollout_truncated()

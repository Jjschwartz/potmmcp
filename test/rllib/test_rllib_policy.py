"""Test RLLib Policy.

Specifically, tests:
- creating a baposgmcp.rllib.RllibPolicy from and rllib.Trainer
- runnning the policy using the baposgmcp runner functions
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

from baposgmcp import runner
import baposgmcp.rllib as ba_rllib
import baposgmcp.stats as stats_lib
import baposgmcp.policy as policy_lib


def _env_creator(config):
    env = posggym.make(config["env_name"])
    env = FlattenObservation(env)
    return RllibMultiAgentEnv(env)


def test_rllib_policy():
    """Run and integration test for RLlibPolicy."""
    env_name = "TwoPaths3x3-v0"
    posggym.make(env_name)

    ray.init()
    register_env(env_name, _env_creator)

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

    for iteration in range(2):
        print(f"== Iteration {iteration} ==")
        result = trainer.train()
        print(pretty_print(result))

    env_model = sample_env.unwrapped.model
    agent_0_policy = policy_lib.RandomPolicy(
        env_model,
        ego_agent=0,
        gamma=0.9
    )
    agent_1_policy = ba_rllib.RllibPolicy(
        env_model,
        ego_agent=1,
        gamma=0.9,
        policy=trainer.get_policy("pi_1"),
        preprocessor=ba_rllib.get_flatten_preprocesor(env_model.obs_spaces[1])
    )
    policies = [agent_0_policy, agent_1_policy]

    logging.basicConfig(level="INFO", format='%(message)s')

    trackers = stats_lib.get_default_trackers(policies)
    renderers = []
    runner.run_sims(
        sample_env.unwrapped,
        policies,
        trackers,
        renderers,
        run_config=runner.RunConfig(seed=0)
    )


if __name__ == "__main__":
    test_rllib_policy()

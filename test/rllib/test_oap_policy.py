import argparse

from pettingzoo.classic import rps_v2

import ray

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.rock_paper_scissors_dummies import (
    AlwaysSameHeuristic
)

import baposgmcp.rllib as ba_rllib


ROCK = 0
ENV_NAME = "RockPaperScissors"


class AlwaysRock(AlwaysSameHeuristic):
    """Always play Rock."""

    def get_initial_state(self):
        return [ROCK]


def _env_creator(config):
    return PettingZooEnv(rps_v2.env())


def _main(args):
    ray.init(num_cpus=1, include_dashboard=False)
    register_env(ENV_NAME, _env_creator)

    if args.use_lstm:
        oap_model_name = ba_rllib.register_lstm_oap_model()
    else:
        oap_model_name = ba_rllib.register_oap_model()

    env_config = {"env_name": ENV_NAME}
    sample_env = _env_creator(env_config)

    agent_ids = list(sample_env.get_agent_ids())
    agent_ids.sort()
    # Note in rllib pettingzoo env all agents have same action and obs spaces
    # https://github.com/ray-project/ray/blob/releases/1.12.0/rllib/env/wrappers/pettingzoo_env.py
    obs_space = sample_env.observation_space
    act_space = sample_env.action_space

    def _policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == agent_ids[0]:
            return "pi_oap"
        return "pi_always_same"

    agent_0_policy_spec = PolicySpec(
        ba_rllib.OAPPPOTorchPolicy,
        obs_space,
        act_space,
        {"oap_loss_coeff": 1.0}
    )
    agent_1_policy_spec = PolicySpec(AlwaysRock, obs_space, act_space, {})

    trainer = ba_rllib.OAPPPOTrainer(
        env=ENV_NAME,
        config={
            "env_config": env_config,
            "gamma": 0.9,
            "num_workers": 0,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 30,   # episode length
            "train_batch_size": 180,
            "metrics_num_episodes_for_smoothing": 100,
            "multiagent": {
                "policies": {
                    "pi_oap": agent_0_policy_spec,
                    "pi_always_same": agent_1_policy_spec
                },
                "policy_mapping_fn": _policy_mapping_fn,
                "policies_to_train": ["pi_oap"],
            },
            "model": {
                "custom_model": oap_model_name,
                "fcnet_hiddens": [64, 32],
                "fcnet_activation": "tanh",
                "use_lstm": False,    # handled by custom model
                "max_seq_len": 30,
                "lstm_cell_size": 64,
                "lstm_use_prev_action": False,
                "lstm_use_prev_reward": False,
                "vf_share_layers": False,
                "custom_model_config": {
                    "oap_share_layers": args.oap_share_layers,
                }
            },
            "num_sgd_iter": 6,
            "use_critic": True,
            "vf_loss_coeff": 1.0,
            "observation_filter": "NoFilter",
            "num_gpus": 0,
            "log_level": "DEBUG",
            "framework": "torch",
        }
    )

    for iteration in range(args.stop_iters):
        print(f"== Iteration {iteration} ==")
        result = trainer.train()
        print(pretty_print(result))

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop_iters", type=int, default=150,
        help="Number of iterations to train."
    )
    parser.add_argument(
        "--oap_share_layers", action="store_false",
        help="Set oap_share_layers=False."
    )
    parser.add_argument(
        "--use_lstm", action="store_true",
        help="Set use_lstm=True."
    )
    _main(parser.parse_args())

"""A script for running K-level reasoning training in Rock, Paper, Scissors.

This implementation uses RLlib, and is based on the following examples:
- https://github.com/ray-project/ray/blob/master/rllib/examples/
  rock_paper_scissors_multiagent.py
- https://github.com/ray-project/ray/blob/master/rllib/examples/
  multi_agent_two_trainers.py

Note for each training iteration only a single level is trained at a time.

"""
import os
import sys
import argparse

try:
    from pettingzoo.classic import rps_v2
except ImportError:
    print(
        "This script requires that the PettingZoo library is installed, with "
        "the 'classic' environment dependencies. For installation instructions"
        " visit https://www.pettingzoo.ml/classic."
    )
    sys.exit(1)

import ray

from ray.rllib.env import PettingZooEnv
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


# pylint: disable=[unused-argument]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--k", type=int, default=1,
    help="Reasoning level (default=1)"
)
parser.add_argument(
    "--stop-iters", type=int, default=150,
    help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000,
    help="Number of timesteps to train."
)


def env_creator(args):
    env = rps_v2.env()
    return env


register_env(
    "RockPaperScissors", lambda config: PettingZooEnv(env_creator(config))
)


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    sample_env = env_creator({})
    obs_space = sample_env.observation_space("player_0")
    act_space = sample_env.action_space("player_0")

    def _policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "player_0":
            return "pi_0"
        return "pi_1"

    policies = []
    trainers = []
    for k in range(args.k+1):
        policies_lk = {    # type: ignore
            "pi_0": (
                RandomPolicy if k == 0 else PPOTorchPolicy,
                obs_space,
                act_space,
                {}
            ),
            "pi_1": (
                PPOTorchPolicy,
                obs_space,
                act_space,
                {}
            )
        }

        trainer_lk = PPOTrainer(
            env="RockPaperScissors",
            config={
                "multiagent": {
                    "policies": policies_lk,
                    "policy_mapping_fn": _policy_mapping_fn,
                    "policies_to_train": ["pi_1"],
                },
                "model": {
                    "vf_share_layers": True,
                },
                "num_sgd_iter": 6,
                "vf_loss_coeff": 0.01,
                "observation_filter": "MeanStdFilter",
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                "framework": "torch",
            },
        )
        policies.append(policies_lk)
        trainers.append(trainer_lk)

    for i in range(args.stop_iters):
        print(f"== Iteration {i} ==")

        for k in range(args.k+1):
            print(f"-- Level {k} --")
            result = trainers[k].train()
            print(pretty_print(result))

        # swap weights of opponent policies
        for k in range(1, args.k+1):
            trainers[k].set_weights(trainers[k-1].get_weights("pi_1"))

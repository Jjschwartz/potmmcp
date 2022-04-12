"""A script for evaluating trained RL policies."""
import argparse

import ray

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from baposgmcp import pbt
import baposgmcp.rllib as ba_rllib

from ray.rllib.agents.ppo import PPOTrainer

from exp_utils import (
    ENV_CONFIG, env_creator, policy_mapping_fn, ENV_NAME
)


def _trainer_make_fn(config):
    return PPOTrainer(env=ENV_NAME, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "policy_dir", type=str,
        help="Path to dir containing trained RL policies"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="Number of evaluation episodes to run."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment."
    )
    args = parser.parse_args()

    ray.init()
    register_env(ENV_NAME, env_creator)

    sample_env = env_creator(ENV_CONFIG)

    eval_config = {
        "render_env": args.render,
        # If True, store videos in this relative directory inside the default
        # output dir (~/ray_results/...)
        # TO ME: I added this here for referent in case I want to use it later
        "record_env": True,
        "evaluation_interval": 1,
        # In episodes (by default)
        # Can be changed to 'timesteps' if desired
        "evaluation_duration": args.num_episodes,
        "evaluation_duration_unit": "episodes",
    }

    print("\n== Importing Graph ==")
    igraph, trainer_map = ba_rllib.import_igraph_trainers(
        igraph_dir=args.policy_dir,
        env_is_symmetric=False,
        trainer_make_fn=_trainer_make_fn,
        trainers_remote=False,
        policy_mapping_fn=policy_mapping_fn,
        extra_config=eval_config
    )
    igraph.display()

    print("== Running Evaluation ==")
    results = {i: {} for i in trainer_map}
    for i, policy_map in trainer_map.items():
        results[i] = {}
        for policy_k_id, trainer in policy_map.items():
            print(f"-- Running Agent ID {i}, Policy {policy_k_id} --")
            results[i][policy_k_id] = trainer.evaluate()

    print("== Evaluation results ==")
    for i, policy_map in results.items():
        for policy_k_id, result in policy_map.items():
            print(f"-- Agent ID {i}, Policy {policy_k_id} --")
            print(pretty_print(result))

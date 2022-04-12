import logging
import argparse
from itertools import product

import ray

from ray.tune.registry import register_env

from ray.rllib.agents.ppo import PPOTrainer

from baposgmcp import runner
import baposgmcp.rllib as ba_rllib
import baposgmcp.stats as stats_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as ba_policy_lib

from exp_utils import (
    ENV_CONFIG, env_creator, policy_mapping_fn, ENV_NAME
)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "policy_dir", type=str,
        help="Path to dir containing trained RL policies"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Experiment seed."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--episode_step_limit", type=int, default=100,
        help="Max steps per episode."
    )
    parser.add_argument(
        "--time_limit", type=int, default=43200,
        help="Experiment time limit, in seconds."
    )
    args = parser.parse_args()

    ray.init()
    register_env(args.env_name, env_creator)

    sample_env = env_creator(ENV_CONFIG)

    print("\n== Importing Graph ==")
    igraph, trainer_map = ba_rllib.import_igraph_trainers(
        igraph_dir=args.policy_dir,
        env_is_symmetric=False,
        trainer_make_fn=_trainer_make_fn,
        trainers_remote=False,
        policy_mapping_fn=policy_mapping_fn
    )

    igraph.display()
    input("Press Any Key to Continue")

    print("\n== Importing Policies ==")
    policy_map = ba_rllib.get_policy_from_trainer_map(trainer_map)

    rllibpolicy_map = {}
    env_model = sample_env.unwrapped.model
    for agent_id, agent_policy_map in policy_map.items():
        rllibpolicy_map[agent_id] = {}
        for policy_id, policy in agent_policy_map.items():
            if "-1" in policy_id:
                new_policy = ba_policy_lib.RandomPolicy(
                    env_model, agent_id, 0.9
                )
            else:
                obs_space = env_model.obs_spaces[int(agent_id)]
                new_policy = ba_rllib.RllibPolicy(
                    env_model,
                    agent_id,
                    0.9,
                    policy,
                    preprocessor=ba_rllib.get_flatten_preprocesor(obs_space)
                )
            rllibpolicy_map[agent_id][policy_id] = new_policy

    print("\n== Running Experiments ==")
    # Run the different ba_rllib.RllibPolicy policies against each other
    logging.basicConfig(level="INFO", format='%(message)s')
    agent_0_policies = list(rllibpolicy_map["0"].values())
    agent_1_policies = list(rllibpolicy_map["1"].values())
    for policies in product(agent_0_policies, agent_1_policies):
        trackers = stats_lib.get_default_trackers(policies)
        renderers = [
            render_lib.EpisodeRenderer(pause_each_step=False)
        ]
        runner.run_sims(
            sample_env.unwrapped,
            policies,
            trackers,
            renderers,
            run_config=runner.RunConfig(
                seed=args.seed,
                num_episodes=args.num_episodes
            )
        )

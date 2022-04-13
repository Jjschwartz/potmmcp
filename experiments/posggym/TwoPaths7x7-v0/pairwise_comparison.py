import logging
import pathlib
import argparse
import os.path as osp
from datetime import datetime
from itertools import product

import ray

from ray.tune.registry import register_env

from ray.rllib.agents.ppo import PPOTrainer

from baposgmcp import runner
import baposgmcp.exp as exp_lib
import baposgmcp.rllib as ba_rllib
import baposgmcp.stats as stats_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as ba_policy_lib

from exp_utils import (
    ENV_CONFIG, env_creator, policy_mapping_fn, ENV_NAME, EXP_RESULTS_DIR
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
        "--seed", type=int, default=0,
        help="Experiment seed."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Experiment time limit, in seconds."
    )
    parser.add_argument(
        "--n_procs", type=int, default=1,
        help="Number of processors/experiments to run in parallel."
    )
    parser.add_argument(
        "--log_level", type=int, default=21,
        help="Experiment log level."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render experiment episodes."
    )
    args = parser.parse_args()

    ray.init()
    register_env(ENV_NAME, env_creator)

    sample_env = env_creator(ENV_CONFIG)

    print("\n== Importing Graph ==")
    igraph, trainer_map = ba_rllib.import_igraph_trainers(
        igraph_dir=args.policy_dir,
        env_is_symmetric=False,
        trainer_make_fn=_trainer_make_fn,
        trainers_remote=False,
        policy_mapping_fn=policy_mapping_fn
    )

    print("\n== Importing Policies ==")
    policy_map = ba_rllib.get_policy_from_trainer_map(trainer_map)

    env_model = sample_env.unwrapped.model

    def _get_rllib_policy_init_fn(pi, agent_id):
        """Get init function for rllib policy.

        This is a little hacky but gets around issure of copying kwargs.
        """
        obs_space = env_model.obs_spaces[int(agent_id)]
        preprocessor = ba_rllib.get_flatten_preprocessor(obs_space)

        def pi_init(model, ego_agent, gamma, **kwargs):
            return ba_rllib.RllibPolicy(
                model=model,
                ego_agent=ego_agent,
                gamma=gamma,
                policy=pi,
                preprocessor=preprocessor,
                **kwargs
            )

        return pi_init

    policy_params_map = {}

    for agent_id, agent_policy_map in policy_map.items():
        policy_params_map[agent_id] = {}
        for policy_id, policy in agent_policy_map.items():
            if "-1" in policy_id:
                policy_params = exp_lib.PolicyParams(
                    name="RandomPolicy",
                    gamma=0.95,
                    kwargs={},
                    init=ba_policy_lib.RandomPolicy
                )
            else:
                obs_space = env_model.obs_spaces[int(agent_id)]
                preprocessor = ba_rllib.get_flatten_preprocessor(obs_space)
                policy_params = exp_lib.PolicyParams(
                    name=f"PPOPolicy_{policy_id}",
                    gamma=0.95,
                    kwargs={},
                    init=_get_rllib_policy_init_fn(policy, agent_id)
                )
            policy_params_map[agent_id][policy_id] = policy_params

    print("\n== Running Experiments ==")
    # Run the different ba_rllib.RllibPolicy policies against each other
    logging.basicConfig(level="INFO", format='%(message)s')

    result_dir = osp.join(EXP_RESULTS_DIR, str(datetime.now()))
    pathlib.Path(result_dir).mkdir(exist_ok=False)

    agent_0_policies = list(policy_params_map["0"].values())
    agent_1_policies = list(policy_params_map["1"].values())
    exp_params_list = []

    renderers = []
    if args.render:
        renderers.append(render_lib.EpisodeRenderer())

    for i, policies in enumerate(product(agent_0_policies, agent_1_policies)):
        exp_params = exp_lib.ExpParams(
            exp_id=i,
            env_name=ENV_NAME,
            policy_params_list=policies,
            run_config=runner.RunConfig(
                seed=args.seed,
                num_episodes=args.num_episodes,
                episode_step_limit=None,
                time_limit=args.time_limit
            ),
            tracker_fn=lambda: stats_lib.get_default_trackers(policies),
            render_fn=lambda: renderers,
        )
        exp_params_list.append(exp_params)

    exp_lib.run_experiments(
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        result_dir=result_dir
    )

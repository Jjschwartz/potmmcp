"""Script for running pairwise evaluation of trained Rllib policies.

The script takes a list of environments names and a list of rllib policy save
directories as arguments. It then runs a pairwise evaluation between each
policy in each of the policy directories for each environment.

"""
import logging
import pathlib
import argparse
import tempfile
import os.path as osp
from datetime import datetime
from typing import Sequence, List
from itertools import combinations_with_replacement, product

from ray.tune.registry import register_env

from baposgmcp import runner
import baposgmcp.exp as exp_lib
import baposgmcp.stats as stats_lib
import baposgmcp.policy as policy_lib
import baposgmcp.render as render_lib

from exp_utils import (
    registered_env_creator,
    EXP_RESULTS_DIR,
    load_agent_policy_params,
    get_base_env
)


def _renderer_fn(**kwargs) -> Sequence[render_lib.Renderer]:
    renderers = []
    if kwargs["render"]:
        renderers.append(render_lib.EpisodeRenderer())
    return renderers


def _tracker_fn(policies: List[policy_lib.BasePolicy],
                **kwargs) -> Sequence[stats_lib.Tracker]:
    trackers = stats_lib.get_default_trackers(policies)
    return trackers


def _get_env_policies_exp_params(env_name: str,
                                 agent_0_policy_dir: str,
                                 agent_1_policy_dir: str,
                                 result_dir: str,
                                 args,
                                 exp_id_init: int) -> List[exp_lib.ExpParams]:
    agent_0_policy_params = load_agent_policy_params(
        agent_0_policy_dir,
        args.gamma,
        env_name,
        include_random_policy=False
    )

    agent_1_policy_params = load_agent_policy_params(
        agent_1_policy_dir,
        args.gamma,
        env_name,
        include_random_policy=False
    )

    renderers = []
    if args.render:
        renderers.append(render_lib.EpisodeRenderer())

    exp_params_list = []
    for i, policies in enumerate(
            product(agent_0_policy_params, agent_1_policy_params)
    ):
        exp_params = exp_lib.ExpParams(
            exp_id=exp_id_init+i,
            env_name=env_name,
            policy_params_list=policies,
            run_config=runner.RunConfig(
                seed=args.seed,
                num_episodes=args.num_episodes,
                episode_step_limit=None,
                time_limit=args.time_limit
            ),
            tracker_fn=_tracker_fn,
            tracker_kwargs={},
            renderer_fn=_renderer_fn,
            renderer_kwargs={"render": args.render},
        )
        exp_params_list.append(exp_params)
    return exp_params_list


def _main(args):
    # check env name is valid
    for env_name in args.env_names:
        get_base_env(env_name, args.seed)
        register_env(env_name, registered_env_creator)

    print("\n== Running Experiments ==")
    logging.basicConfig(level=args.log_level, format='%(message)s')
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir_name = f"pairwise_comparison_{timestr}"
    result_dir = tempfile.mkdtemp(prefix=result_dir_name, dir=EXP_RESULTS_DIR)
    exp_lib.write_experiment_arguments(vars(args), result_dir)

    exp_params_list = []
    # since env is symmetric we only need every combination of policies
    # rather than the full product of the policy spaces
    policy_combos = list(combinations_with_replacement(args.policy_dirs, 2))
    for env_name in args.env_names:
        for (agent_0_policy_dir, agent_1_policy_dir) in policy_combos:
            exp_params_list.extend(
                _get_env_policies_exp_params(
                    env_name,
                    agent_0_policy_dir,
                    agent_1_policy_dir,
                    result_dir,
                    args,
                    exp_id_init=len(exp_params_list)
                )
            )

    print(f"== Running {len(exp_params_list)} Experiments ==")
    exp_lib.run_experiments(
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        result_dir=result_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_names", type=str, nargs="+",
        help="Name of the environments to train on."
    )
    parser.add_argument(
        "--policy_dirs", type=str, nargs="+",
        help="Paths to dirs containing trained RL policies"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Experiment seed."
    )
    parser.add_argument(
        "--gamma", type=int, default=0.99,
        help="Discount hyperparam."
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
    _main(parser.parse_args())

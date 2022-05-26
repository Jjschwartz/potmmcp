import logging
import pathlib
import argparse
import os.path as osp
from datetime import datetime
from itertools import product
from typing import Sequence, List

from ray.tune.registry import register_env

from baposgmcp import runner
import baposgmcp.exp as exp_lib
import baposgmcp.tree as tree_lib
import baposgmcp.stats as stats_lib
import baposgmcp.policy as policy_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as ba_policy_lib

from exp_utils import (
    EXP_RESULTS_DIR,
    registered_env_creator,
    get_base_env,
    load_agent_policy_params,
    load_agent_policies,
    load_agent_policy
)


def _baposgmcp_init_fn(model, ego_agent, gamma, **kwargs):
    """Get BAPOSGMCP init function.

    This function which handles dynamic loading of other agent policies.
    This is needed to ensure independent policies are used for each experiment
    when running experiments in parallel.
    """
    args = kwargs.pop("args")
    env_name = kwargs.pop("env_name")
    other_agent_policy_dir = kwargs.pop("other_agent_policy_dir")

    if "rollout_policy_id" in kwargs:
        rollout_policy_id = kwargs.pop("rollout_policy_id")
        rollout_policy_dir = kwargs.pop("rollout_policy_dir")
    else:
        rollout_policy_id = None
        rollout_policy_dir = ""

    other_agent_id = (ego_agent + 1) % 2
    other_policies = {
        other_agent_id: load_agent_policies(
            other_agent_id,
            env_name,
            other_agent_policy_dir,
            gamma,
            include_random_policy=False,
            env_seed=args.seed
        )
    }

    if rollout_policy_id is None:
        rollout_policy = ba_policy_lib.RandomPolicy(
            model, ego_agent, gamma
        )
    else:
        rollout_policy = load_agent_policy(
            rollout_policy_dir,
            rollout_policy_id,
            ego_agent,
            env_name,
            gamma,
            args.seed
        )

    return tree_lib.BAPOSGMCP(
        model,
        ego_agent,
        gamma,
        other_policies=other_policies,
        rollout_policy=rollout_policy,
        **kwargs
    )


def _renderer_fn(**kwargs) -> Sequence[render_lib.Renderer]:
    renderers = []
    if kwargs["render"]:
        renderers.append(render_lib.EpisodeRenderer())
        renderers.append(render_lib.PolicyBeliefRenderer())
    return renderers


def _tracker_fn(policies: List[policy_lib.BasePolicy],
                **kwargs) -> Sequence[stats_lib.Tracker]:
    trackers = stats_lib.get_default_trackers(policies)
    trackers.append(stats_lib.BayesAccuracyTracker(2))
    return trackers


def _get_env_policies_exp_params(env_name: str,
                                 baposgmcp_policy_dir: str,
                                 other_agent_policy_dir: str,
                                 result_dir: str,
                                 args,
                                 exp_id_init: int) -> List[exp_lib.ExpParams]:
    """Get exp params for given env and policy groups.

    Assumes rollout policy for baposgmcp is in baposgmcp_policy_dir.

    baposgmcp_policy_dir specifies the directory to load policies from that are
    used within BAPOSGMCP for the other agent policies.

    other_agent_policy_dir specifiec the directory to load policies from that
    are used as the other agent policies during testing.
    """
    print("== Creating Experiments ==")
    sample_env = get_base_env(env_name, args.seed)
    env_model = sample_env.model

    # env is symmetric so only need to run BAPOSGMCP for a single agent
    baposgmcp_agent_id = 0

    exp_params_list = []
    exp_id = exp_id_init
    for num_sims in args.num_sims:
        baposgmcp_params = exp_lib.PolicyParams(
            name=f"BAPOSGMCP_{baposgmcp_agent_id}",
            gamma=args.gamma,
            kwargs={
                "other_policy_prior": None,     # uniform
                "num_sims": num_sims,
                "c_init": 1.0,
                "c_base": 100.0,
                "truncated": True,
                "reinvigorator": tree_lib.BABeliefRejectionSampler(env_model),
                "extra_particles_prop": 1.0 / 16,
                "step_limit": sample_env.spec.max_episode_steps,
                "epsilon": 0.01,
                "policy_id": "pi_baposgmcp",
                # The following are needed for init fn and are removed by
                # custom init fn
                "args": args,
                "env_name": env_name,
                "other_agent_policy_dir": baposgmcp_policy_dir,
                "rollout_policy_id": args.rollout_policy_id,
                "rollout_policy_dir": baposgmcp_policy_dir
            },
            init=_baposgmcp_init_fn,
            info={
                "other_agent_policy_dir": baposgmcp_policy_dir,
                "rollout_policy_id": args.rollout_policy_id,
                "rollout_policy_dir": baposgmcp_policy_dir
            }
        )

        other_agent_policy_params = load_agent_policy_params(
            other_agent_policy_dir,
            args.gamma,
            env_name,
            include_random_policy=False
        )

        for policy_params in other_agent_policy_params:
            policies = [baposgmcp_params, policy_params]

            exp_params = exp_lib.ExpParams(
                exp_id=exp_id,
                env_name=env_name,
                policy_params_list=policies,
                run_config=runner.RunConfig(
                    seed=args.seed,
                    num_episodes=args.num_episodes,
                    episode_step_limit=args.episode_step_limit,
                    time_limit=args.time_limit
                ),
                tracker_fn=_tracker_fn,
                tracker_kwargs={},
                renderer_fn=_renderer_fn,
                renderer_kwargs={"render": args.render}
            )

            exp_params_list.append(exp_params)
            exp_id += 1

            if args.debug and exp_id-exp_id_init == args.n_procs:
                break
        if args.debug and exp_id-exp_id_init == args.n_procs:
            break

    return exp_params_list


def _main(args):
    # check env name is valid
    for env_name in args.env_names:
        get_base_env(env_name, args.seed)
        register_env(env_name, registered_env_creator)

    print("== Running Experiments ==")
    logging.basicConfig(level="INFO", format='%(message)s')
    result_dir = osp.join(EXP_RESULTS_DIR, str(datetime.now()))
    pathlib.Path(result_dir).mkdir(exist_ok=False)

    exp_params_list = []
    for env_name in args.env_names:
        for (baposgmcp_policy_dir, other_agent_policy_dir) in product(
                args.baposgmcp_policy_dirs, args.other_agent_policy_dirs
        ):
            exp_params = _get_env_policies_exp_params(
                env_name,
                baposgmcp_policy_dir,
                other_agent_policy_dir,
                result_dir,
                args,
                exp_id_init=len(exp_params_list)
            )
            exp_params_list.extend(exp_params)

    print(f"== Running {len(exp_params_list)} Experiments ==")
    if args.debug:
        input("In DEBUG mode. Continue?")

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
        "--baposgmcp_policy_dirs", type=str, nargs="+",
        help="Paths to dirs containing trained RL policies for BAPOSGMCP"
    )
    parser.add_argument(
        "--other_agent_policy_dirs", type=str, nargs="+",
        help="Paths to dirs containing trained RL policies to test against"
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
        "--num_episodes", type=int, default=1,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--episode_step_limit", type=int, default=None,
        help=(
            "Episode step limit. If None then uses default step limit for the "
            "env."
        )
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Experiment time limit, in seconds."
    )
    parser.add_argument(
        "--num_sims", type=int, nargs="*", default=[128],
        help="Number of simulations per search."
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
    parser.add_argument(
        "--rollout_policy_id", type=str, default="None",
        help="ID of policy to use for BAPOSGMCP rollouts, if None use random."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run debug experiment (runs only a single pairing)."
    )
    _main(parser.parse_args())

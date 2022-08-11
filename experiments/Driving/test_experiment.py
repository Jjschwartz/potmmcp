import logging
import argparse
from pprint import pprint
from itertools import product
from typing import Sequence, List

from ray.tune.registry import register_env

import baposgmcp.run as run_lib
import baposgmcp.tree as tree_lib
import baposgmcp.policy as policy_lib

from render import PositionBeliefRenderer
from exp_utils import (
    registered_env_creator,
    get_base_env,
    load_agent_policy_params,
    load_agent_policies,
    get_result_dir
)


def _baposgmcp_init_fn(model, ego_agent, gamma, **kwargs):
    """Get BAPOSGMCP init function.

    This function which handles dynamic loading of other agent and
    rollout policies. This is required to ensure independent policies are used
    for each experiment when running experiments in parallel.
    """
    env_name = kwargs.pop("env_name")
    seed = kwargs.pop("seed")
    other_agent_policy_dir = kwargs.pop("other_agent_policy_dir")
    other_agent_policy_ids = kwargs.pop("other_agent_policy_ids")

    if "rollout_policy_ids" in kwargs:
        rollout_policy_ids = kwargs.pop("rollout_policy_ids")
        rollout_policy_dir = kwargs.pop("rollout_policy_dir")
    else:
        rollout_policy_ids = None
        rollout_policy_dir = ""

    other_agent_id = (ego_agent + 1) % 2
    other_policies = {
        other_agent_id: load_agent_policies(
            other_agent_id,
            env_name,
            other_agent_policy_dir,
            gamma,
            include_random_policy=False,
            env_seed=seed,
            include_policy_ids=other_agent_policy_ids
        )
    }

    if rollout_policy_ids is None:
        rollout_policies = {
            "pi_-1": policy_lib.RandomPolicy(model, ego_agent, gamma)
        }
        rollout_selection = {
            pi_id: "pi_-1" for pi_id in other_policies[other_agent_id]
        }
    else:
        rollout_policies = load_agent_policies(
            ego_agent,
            env_name,
            rollout_policy_dir,
            gamma,
            include_random_policy=False,
            env_seed=seed,
            include_policy_ids=rollout_policy_ids
        )
        rollout_selection = {}
        for pi_id in other_policies[other_agent_id]:
            # Assumed policies of the form 'pi_k'
            other_k = int(pi_id.split("_")[-1])
            ro_pi_id = f"pi_{other_k+1}"
            assert ro_pi_id in rollout_policies, f"{ro_pi_id} invalid"
            rollout_selection[pi_id] = ro_pi_id

    return tree_lib.BAPOSGMCP(
        model,
        ego_agent,
        gamma,
        other_policies=other_policies,
        rollout_policies=rollout_policies,
        rollout_selection=rollout_selection,
        **kwargs
    )


def _renderer_fn(**kwargs) -> Sequence[run_lib.Renderer]:
    renderers = []
    if kwargs["render"]:
        renderers = [
            run_lib.EpisodeRenderer(),
            # run_lib.SearchTreeRenderer(1),
            # PositionBeliefRenderer(),
            run_lib.PauseRenderer()
        ]
    return renderers


def _tracker_fn(policies: List[policy_lib.BasePolicy],
                **kwargs) -> Sequence[run_lib.Tracker]:
    trackers = run_lib.get_default_trackers(policies)

    tracker_kwargs = {
        "num_agents": len(policies),
        "track_per_step": True,
        "step_limit": kwargs["step_limit"]
    }

    trackers.append(run_lib.BayesAccuracyTracker(**tracker_kwargs))
    # trackers.append(
    #     run_lib.ActionDistributionDistanceTracker(**tracker_kwargs)
    # )
    # trackers.append(run_lib.BeliefHistoryAccuracyTracker(**tracker_kwargs))
    trackers.append(run_lib.BeliefStateAccuracyTracker(**tracker_kwargs))
    return trackers


def _get_env_policies_exp_params(env_name: str,
                                 baposgmcp_policy_dir: str,
                                 other_agent_policy_dir: str,
                                 result_dir: str,
                                 args,
                                 exp_id_init: int) -> List[run_lib.ExpParams]:
    """Get exp params for given env and policy groups.

    Assumes rollout policy for baposgmcp is in baposgmcp_policy_dir.

    baposgmcp_policy_dir specifies the directory to load policies from that are
    used within BAPOSGMCP for the other agent policies.

    other_agent_policy_dir specifiec the directory to load policies from that
    are used as the other agent policies during testing.
    """
    sample_env = get_base_env(env_name, args.init_seed)
    env_model = sample_env.model

    episode_step_limit = sample_env.spec.max_episode_steps
    if episode_step_limit is None:
        episode_step_limit = args.episode_step_limit
    elif args.episode_step_limit is not None:
        episode_step_limit = min(episode_step_limit, args.episode_step_limit)

    # env is symmetric so only need to run BAPOSGMCP for a single agent
    baposgmcp_agent_id = 0

    exp_params_list = []
    exp_id = exp_id_init
    for exp_seed, num_sims in product(range(args.num_seeds), args.num_sims):
        baposgmcp_params = run_lib.PolicyParams(
            name=f"BAPOSGMCP_{baposgmcp_agent_id}",
            gamma=args.gamma,
            kwargs={
                "other_policy_prior": None,     # uniform
                "num_sims": num_sims,
                "c_init": 1.25,
                "c_base": 20000.0,
                "truncated": True,
                "reinvigorator": tree_lib.BABeliefRejectionSampler(env_model),
                "extra_particles_prop": 1.0 / 16,
                "step_limit": episode_step_limit,
                "epsilon": 0.01,
                "policy_id": "pi_baposgmcp",
                # The following are needed for init fn and are removed by
                # custom init fn
                "env_name": env_name,
                "seed": args.init_seed + exp_seed,
                "other_agent_policy_dir": baposgmcp_policy_dir,
                "other_agent_policy_ids": args.baposgmcp_other_policy_ids,
                "rollout_policy_ids": args.rollout_policy_ids,
                "rollout_policy_dir": baposgmcp_policy_dir
            },
            init=_baposgmcp_init_fn,
            info={
                "other_agent_policy_dir": baposgmcp_policy_dir,
                "other_agent_policy_ids": args.baposgmcp_other_policy_ids,
                "rollout_policy_ids": args.rollout_policy_ids,
                "rollout_policy_dir": baposgmcp_policy_dir,
                "seed": args.init_seed + exp_seed
            }
        )

        other_agent_policy_params = load_agent_policy_params(
            other_agent_policy_dir,
            args.gamma,
            env_name,
            include_random_policy=False,
            include_policy_ids=args.exp_other_policy_ids
        )

        for policy_params in other_agent_policy_params:
            policies = [baposgmcp_params, policy_params]

            exp_params = run_lib.ExpParams(
                exp_id=exp_id,
                env_name=env_name,
                policy_params_list=policies,
                run_config=run_lib.RunConfig(
                    seed=args.init_seed + exp_seed,
                    num_episodes=args.num_episodes,
                    episode_step_limit=episode_step_limit,
                    time_limit=args.time_limit,
                    use_checkpointing=True
                ),
                tracker_fn=_tracker_fn,
                tracker_kwargs={"step_limit": episode_step_limit},
                renderer_fn=_renderer_fn,
                renderer_kwargs={"render": args.render},
                record_env=args.record_env,
                record_env_freq=max(1, args.num_episodes // 10)
            )

            exp_params_list.append(exp_params)
            exp_id += 1

    return exp_params_list


def _main(args):
    # check env name is valid
    for env_name in args.env_names:
        get_base_env(env_name, args.init_seed)
        register_env(env_name, registered_env_creator)

    print("== Running Experiments ==")
    print("Experiment arguments:")
    pprint(vars(args))
    logging.basicConfig(
        level=args.log_level,
        # [Day-Month Hour-Minute-Second] Message
        format='[%(asctime)s] %(message)s', datefmt='%d-%m %H:%M:%S'
    )
    num_sim_str = "nsims" + "_".join([str(n) for n in args.num_sims])
    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    result_dir_name_prefix = f"experiment_{num_sim_str}_{seed_str}"
    result_dir = get_result_dir(result_dir_name_prefix, args.root_save_dir)
    run_lib.write_experiment_arguments(vars(args), result_dir)

    print("== Creating Experiments ==")
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
    print(f"== Using {args.n_procs} CPUs ==")
    run_lib.run_experiments(
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        result_dir=result_dir
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--init_seed", type=int, default=0,
        help="Experiment start seed."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--num_sims", type=int, nargs="*", default=[128],
        help="Number of simulations per search."
    )
    parser.add_argument(
        "--log_level", type=int, default=18,
        help="Experiment log level."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render experiment episodes."
    )
    parser.add_argument(
        "--record_env", action="store_true",
        help="Record renderings of experiment episodes."
    )
    args = parser.parse_args()

    args.env_names = ["Driving7x7RoundAbout-v0"]
    args.rollout_policy_ids = ["pi_0", "pi_1", "pi_2", "pi_3", "pi_4"]
    args.baposgmcp_other_policy_ids = ["pi_0", "pi_1", "pi_2", "pi_3"]
    args.exp_other_policy_ids = ["pi_3"]
    args.baposgmcp_policy_dirs = [
        "/home/jonathon/baposgmcp_results/Driving/rl_policies/2022-08-02_banrmcp/train_klr_Driving7x7RoundAbout-v0_k4_seed0_2022-08-03_01-31-491db54_mq"
    ]
    args.other_agent_policy_dirs = [
        "/home/jonathon/baposgmcp_results/Driving/rl_policies/2022-08-02_banrmcp/train_klr_Driving7x7RoundAbout-v0_k4_seed0_2022-08-03_01-31-491db54_mq"
    ]
    args.episode_step_limit = None
    args.gamma = 0.99
    args.num_seeds = 1
    args.time_limit = None
    args.n_procs = 1
    args.root_save_dir = "/tmp"

    _main(args)

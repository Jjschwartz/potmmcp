import pathlib
import argparse
import os.path as osp
from datetime import datetime

import ray

from ray.tune.registry import register_env

from ray.rllib.agents.ppo import PPOTrainer

from baposgmcp import runner
import baposgmcp.exp as exp_lib
import baposgmcp.tree as tree_lib
import baposgmcp.rllib as ba_rllib
import baposgmcp.stats as stats_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as ba_policy_lib

from exp_utils import (
    ENV_CONFIG, env_creator, policy_mapping_fn, ENV_NAME, EXP_RESULTS_DIR
)


def _trainer_make_fn(config):
    return PPOTrainer(env=ENV_NAME, config=config)


def _import_policies(args):
    igraph, trainer_map = ba_rllib.import_igraph_trainers(
        igraph_dir=args.policy_dir,
        env_is_symmetric=False,
        trainer_make_fn=_trainer_make_fn,
        trainers_remote=False,
        policy_mapping_fn=policy_mapping_fn,
        extra_config={
            "num_gpus": args.gpu_utilization
        }
    )
    policy_map = ba_rllib.get_policy_from_trainer_map(trainer_map)
    return policy_map


def _load_policy_params(args):
    print("\n== Importing Policy Params ==")
    sample_env = env_creator(ENV_CONFIG)
    env_model = sample_env.unwrapped.model

    policy_map = _import_policies(args)

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
        random_policy_added = False
        for policy_id, policy in agent_policy_map.items():
            if "-1" in policy_id:
                policy_params = exp_lib.PolicyParams(
                    name="RandomPolicy",
                    gamma=0.95,
                    kwargs={"policy_id": policy_id},
                    init=ba_policy_lib.RandomPolicy
                )
                random_policy_added = True
            else:
                policy_params = exp_lib.PolicyParams(
                    name=f"PPOPolicy_{policy_id}",
                    gamma=0.95,
                    kwargs={"policy_id": policy_id},
                    init=_get_rllib_policy_init_fn(policy, agent_id)
                )
            policy_params_map[agent_id][policy_id] = policy_params

        if not random_policy_added:
            policy_params = exp_lib.PolicyParams(
                name="RandomPolicy",
                gamma=0.95,
                kwargs={"policy_id": f"pi_-1_{agent_id}"},
                init=ba_policy_lib.RandomPolicy
            )
            policy_params_map[agent_id][f"pi_-1_{agent_id}"] = policy_params

    return policy_params_map


def _load_policies(args):
    print("\n== Importing Policies ==")
    sample_env = env_creator(ENV_CONFIG)
    env_model = sample_env.unwrapped.model

    rllib_policy_map = _import_policies(args)

    policies_map = {}
    for agent_id, agent_policy_map in rllib_policy_map.items():
        policies_map[agent_id] = {}
        random_policy_added = False
        for policy_id, policy in agent_policy_map.items():
            if "-1" in policy_id:
                new_policy = ba_policy_lib.RandomPolicy(
                    env_model, int(agent_id), 0.95
                )
                random_policy_added = True
            else:
                obs_space = env_model.obs_spaces[int(agent_id)]
                preprocessor = ba_rllib.get_flatten_preprocessor(obs_space)
                new_policy = ba_rllib.RllibPolicy(
                    model=env_model,
                    ego_agent=int(agent_id),
                    gamma=0.95,
                    policy=policy,
                    policy_id=policy_id,
                    preprocessor=preprocessor,
                )
            policies_map[agent_id][policy_id] = new_policy

        if not random_policy_added:
            new_policy = ba_policy_lib.RandomPolicy(
                env_model,
                int(agent_id),
                0.95,
                policy_id=f"pi_-1_{agent_id}"
            )
            policies_map[agent_id][f"pi_-1_{agent_id}"] = new_policy

    return policies_map


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
        "--num_episodes", type=int, default=1,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Experiment time limit, in seconds."
    )
    parser.add_argument(
        "--num_sims", type=int, default=128,
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
        "-gpu", "--gpu_utilization", type=float, default=0.9,
        help="Proportion of availabel GPU to use."
    )
    args = parser.parse_args()

    ray.init()
    register_env(ENV_NAME, env_creator)

    policy_params_map = _load_policy_params(args)
    policies_map = _load_policies(args)

    print("\n== Creating Experiments ==")
    # Experiments: Run BAPOSGMCP against each RLLib policy for both agents
    result_dir = osp.join(EXP_RESULTS_DIR, str(datetime.now()))
    pathlib.Path(result_dir).mkdir(exist_ok=False)

    sample_env = env_creator(ENV_CONFIG)
    env_model = sample_env.unwrapped.model

    exp_params_list = []
    exp_id = 0
    for agent_id in [0, 1]:
        other_agent_id = (agent_id + 1) % 2

        other_agent_policy_params = list(
            policy_params_map[str(other_agent_id)].values()
        )
        other_policies = {
            other_agent_id: policies_map[str(other_agent_id)]
        }

        baposgmcp_params = exp_lib.PolicyParams(
            name=f"BAPOSGMCP_{agent_id}",
            gamma=0.95,
            kwargs={
                "other_policies": other_policies,
                "other_policy_prior": None,     # uniform
                "num_sims": args.num_sims,
                "rollout_policy": ba_policy_lib.RandomRolloutPolicy(
                    env_model, agent_id, 0.95
                ),
                "uct_c": 10.0,
                "reinvigorator": tree_lib.BABeliefRejectionSampler(env_model),
            },
            init=tree_lib.BAPOSGMCP
        )

        renderers = []
        if args.render:
            renderers.append(render_lib.EpisodeRenderer())

        for policy_params in other_agent_policy_params:
            if agent_id == 0:
                policies = [baposgmcp_params, policy_params]
            else:
                policies = [policy_params, baposgmcp_params]

            trackers = stats_lib.get_default_trackers(policies)
            trackers.append(stats_lib.BayesAccuracyTracker(2))

            exp_params = exp_lib.ExpParams(
                exp_id=exp_id,
                env_name=ENV_NAME,
                policy_params_list=policies,
                run_config=runner.RunConfig(
                    seed=args.seed,
                    num_episodes=args.num_episodes,
                    episode_step_limit=None,
                    time_limit=args.time_limit
                ),
                tracker_fn=lambda: trackers,
                render_fn=lambda: renderers,
            )

            exp_params_list.append(exp_params)
            exp_id += 1

            break
        break

    print("\n== Running Experiments ==")
    exp_lib.run_experiments(
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        result_dir=result_dir
    )

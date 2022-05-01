import pathlib
import argparse
import os.path as osp
from datetime import datetime

import ray

from ray.tune.registry import register_env

from ray.rllib.agents.ppo import PPOTrainer

from baposgmcp import pbt
from baposgmcp import runner
import baposgmcp.exp as exp_lib
import baposgmcp.tree as tree_lib
import baposgmcp.rllib as ba_rllib
import baposgmcp.stats as stats_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as ba_policy_lib

from exp_utils import EXP_RESULTS_DIR, registered_env_creator, get_base_env


def _trainer_make_fn(config):
    return PPOTrainer(env=config["env_config"]["env_name"], config=config)


def _import_rllib_policies(args):
    print("\n== Importing Graph ==")
    igraph, trainer_map = ba_rllib.import_igraph_trainers(
        igraph_dir=args.policy_dir,
        env_is_symmetric=True,
        trainer_make_fn=_trainer_make_fn,
        trainers_remote=False,
        policy_mapping_fn=None,
        extra_config={
            # only using policies for rollouts so no GPU
            "num_gpus": 0.0
        }
    )
    print("\n== Importing Policies ==")
    policy_map = ba_rllib.get_policy_from_trainer_map(trainer_map)
    return policy_map


def _load_agent_policy_params(args):
    print("\n== Importing Policy Params ==")
    sample_env = get_base_env(args)
    env_model = sample_env.model
    policy_map = _import_rllib_policies(args)

    def _get_rllib_policy_init_fn(pi):
        """Get init function for rllib policy.

        This is a little hacky but gets around issure of copying kwargs.
        """
        obs_space = env_model.obs_spaces[0]
        preprocessor = ba_rllib.get_flatten_preprocessor(obs_space)

        def pi_init(model, ego_agent, gamma, **kwargs):
            return ba_rllib.PPORllibPolicy(
                model=model,
                ego_agent=ego_agent,
                gamma=gamma,
                policy=pi,
                preprocessor=preprocessor,
                **kwargs
            )

        return pi_init

    policy_params_map = {}
    symmetric_agent_id = pbt.InteractionGraph.SYMMETRIC_ID
    random_policy_added = False
    for policy_id, policy in policy_map[symmetric_agent_id].items():
        if "-1" in policy_id:
            policy_params = exp_lib.PolicyParams(
                name="RandomPolicy",
                gamma=args.gamma,
                kwargs={"policy_id": policy_id},
                init=ba_policy_lib.RandomPolicy
            )
            random_policy_added = True
        else:
            policy_params = exp_lib.PolicyParams(
                name=f"PPOPolicy_{policy_id}",
                gamma=args.gamma,
                kwargs={"policy_id": policy_id},
                init=_get_rllib_policy_init_fn(policy)
            )
        policy_params_map[policy_id] = policy_params

    if not random_policy_added:
        policy_params = exp_lib.PolicyParams(
            name="RandomPolicy",
            gamma=args.gamma,
            kwargs={"policy_id": "pi_-1"},
            init=ba_policy_lib.RandomPolicy
        )
        policy_params_map["pi_-1"] = policy_params

    return policy_params_map


def _load_policy_params(args):
    env = get_base_env(args)
    # Need to load seperate policies for each agent to ensure same policy
    # object is not used for two agents at the same time
    policy_params_map = {
        i: _load_agent_policy_params(args) for i in range(env.n_agents)
    }
    return policy_params_map


def _load_agent_policies(args, agent_id: int):
    print("\n== Importing Policies ==")
    sample_env = get_base_env(args)
    env_model = sample_env.unwrapped.model

    policy_map = _import_rllib_policies(args)

    policies_map = {}
    random_policy_added = False
    symmetric_agent_id = pbt.InteractionGraph.SYMMETRIC_ID
    for policy_id, policy in policy_map[symmetric_agent_id].items():
        if "-1" in policy_id:
            new_policy = ba_policy_lib.RandomPolicy(
                env_model, agent_id, args.gamma
            )
            random_policy_added = True
        else:
            obs_space = env_model.obs_spaces[agent_id]
            preprocessor = ba_rllib.get_flatten_preprocessor(obs_space)
            new_policy = ba_rllib.PPORllibPolicy(
                model=env_model,
                ego_agent=agent_id,
                gamma=args.gamma,
                policy=policy,
                policy_id=policy_id,
                preprocessor=preprocessor,
            )
        policies_map[policy_id] = new_policy

    if not random_policy_added:
        new_policy = ba_policy_lib.RandomPolicy(
            env_model,
            agent_id,
            args.gamma,
            policy_id="pi_-1"
        )
        policies_map["pi_-1"] = new_policy

    return policies_map


def _load_agent_policy(args, policy_id, agent_id):
    agent_policies = _load_agent_policies(args, agent_id)
    return agent_policies[policy_id]


def _load_policies(args):
    env = get_base_env(args)
    # Need to load seperate policies for each agent to ensure same policy
    # object is not used for two agents at the same time
    policy_params_map = {
        i: _load_agent_policies(args, i) for i in range(env.n_agents)
    }
    return policy_params_map


def _create_exp_params(args, policy_params_map, policies_map):
    print("\n== Creating Experiments ==")
    sample_env = get_base_env(args)
    env_model = sample_env.model

    exp_params_list = []
    exp_id = 0
    for agent_id in range(env_model.n_agents):
        other_agent_id = (agent_id + 1) % 2

        other_agent_policy_params = list(
            policy_params_map[other_agent_id].values()
        )
        other_policies = {
            other_agent_id: policies_map[other_agent_id]
        }
        if args.rollout_policy_id is None:
            rollout_policy = ba_policy_lib.RandomPolicy(
                env_model, agent_id, 0.99
            )
        else:
            rollout_policy = _load_agent_policy(
                args, args.rollout_policy_id, agent_id
            )

        baposgmcp_params = exp_lib.PolicyParams(
            name=f"BAPOSGMCP_{agent_id}",
            gamma=args.gamma,
            kwargs={
                "other_policies": other_policies,
                "other_policy_prior": None,     # uniform
                "num_sims": args.num_sims,
                "rollout_policy": rollout_policy,
                "c_init": 1.0,
                "c_base": 100.0,
                "truncated": True,
                "reinvigorator": tree_lib.BABeliefRejectionSampler(env_model),
                "extra_particles_prop": 1.0 / 16,
                "step_limit": sample_env.spec.max_episode_steps,
                "epsilon": 0.01
            },
            init=tree_lib.BAPOSGMCP
        )

        renderers = []
        if args.render:
            renderers.append(render_lib.EpisodeRenderer())
            renderers.append(render_lib.PolicyBeliefRenderer())

        for policy_params in other_agent_policy_params:
            if agent_id == 0:
                policies = [baposgmcp_params, policy_params]
            else:
                policies = [policy_params, baposgmcp_params]

            trackers = stats_lib.get_default_trackers(policies)
            trackers.append(stats_lib.BayesAccuracyTracker(2))

            exp_params = exp_lib.ExpParams(
                exp_id=exp_id,
                env_name=args.env_name,
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

            if args.debug:
                break
        if args.debug:
            break

    return exp_params_list


def _main(args):
    # check env name is valid
    get_base_env(args)

    ray.init()
    register_env(args.env_name, registered_env_creator)

    policy_params_map = _load_policy_params(args)
    policies_map = _load_policies(args)
    exp_params_list = _create_exp_params(args, policy_params_map, policies_map)

    # Experiments: Run BAPOSGMCP against each RLLib policy for both agents
    result_dir = osp.join(EXP_RESULTS_DIR, str(datetime.now()))
    pathlib.Path(result_dir).mkdir(exist_ok=False)

    print("\n== Running Experiments ==")
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
        "env_name", type=str,
        help="Name of the environment to train on."
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
        "--gamma", type=int, default=0.99,
        help="Discount hyperparam."
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
        "--rollout_policy_id", type=str, default="None",
        help="ID of policy to use for BAPOSGMCP rollouts, if None use random."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run debug experiment (runs only a single pairing)."
    )
    _main(parser.parse_args())

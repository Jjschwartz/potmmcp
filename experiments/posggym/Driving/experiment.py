import pathlib
import argparse
import os.path as osp
from datetime import datetime
from typing import Sequence, List

from ray.tune.registry import register_env

from ray.rllib.agents.ppo import PPOTrainer

from baposgmcp import pbt
from baposgmcp import runner
import baposgmcp.exp as exp_lib
import baposgmcp.tree as tree_lib
import baposgmcp.rllib as ba_rllib
import baposgmcp.stats as stats_lib
import baposgmcp.policy as policy_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as ba_policy_lib

from exp_utils import EXP_RESULTS_DIR, registered_env_creator, get_base_env


def _trainer_make_fn(config):
    return PPOTrainer(env=config["env_config"]["env_name"], config=config)


def _import_rllib_policy(policy_dir, policy_id, agent_id):
    return ba_rllib.import_policy(
        policy_id=policy_id,
        igraph_dir=policy_dir,
        env_is_symmetric=True,
        agent_id=agent_id,
        trainer_make_fn=_trainer_make_fn,
        policy_mapping_fn=None,
        trainers_remote=False,
        extra_config={
            # only using policies for inference on single obs so no GPU
            "num_gpus": 0.0,
            # no need for seperate rollout workers either
            "num_workers": 0
        }
    )


def _rllib_policy_init_fn(model, ego_agent, gamma, **kwargs):
    policy_dir = kwargs.pop("policy_dir")
    preprocessor = ba_rllib.get_flatten_preprocessor(
        model.obs_spaces[ego_agent]
    )
    pi = _import_rllib_policy(policy_dir, kwargs["policy_id"], ego_agent)

    return ba_rllib.PPORllibPolicy(
        model=model,
        ego_agent=ego_agent,
        gamma=gamma,
        policy=pi,
        preprocessor=preprocessor,
        **kwargs
    )


def _load_agent_policy_params(args):
    print("== Loading Policy Params ==")
    igraph = ba_rllib.import_igraph(args.policy_dir, True)

    policy_params_list = []
    random_policy_added = False
    for policy_id in igraph.policies[pbt.InteractionGraph.SYMMETRIC_ID]:
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
                kwargs={
                    "policy_dir": args.policy_dir,
                    "policy_id": policy_id
                },
                init=_rllib_policy_init_fn
            )
        policy_params_list.append(policy_params)

    if not random_policy_added:
        policy_params = exp_lib.PolicyParams(
            name="RandomPolicy",
            gamma=args.gamma,
            kwargs={"policy_id": "pi_-1"},
            init=ba_policy_lib.RandomPolicy
        )
        policy_params_list.append(policy_params)

    return policy_params_list


def _load_agent_policies(args, agent_id: int):
    print(f"== Loading Policies for Agent {agent_id} ==")
    sample_env = get_base_env(args)
    env_model = sample_env.unwrapped.model

    _, policy_map = ba_rllib.import_igraph_policies(
        igraph_dir=args.policy_dir,
        env_is_symmetric=True,
        trainer_make_fn=_trainer_make_fn,
        trainers_remote=False,
        policy_mapping_fn=None,
        extra_config={
            # only using policies for rollouts so no GPU
            "num_gpus": 0.0,
            # no need for seperate rollout workers either
            "num_workers": 0
        }
    )

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
    rllib_policy = _import_rllib_policy(args.policy_dir, policy_id, agent_id)

    sample_env = get_base_env(args)
    env_model = sample_env.unwrapped.model
    obs_space = env_model.obs_spaces[agent_id]
    preprocessor = ba_rllib.get_flatten_preprocessor(obs_space)

    policy = ba_rllib.PPORllibPolicy(
        model=env_model,
        ego_agent=agent_id,
        gamma=args.gamma,
        policy=rllib_policy,
        policy_id=policy_id,
        preprocessor=preprocessor,
    )
    return policy


def _baposgmcp_init_fn(model, ego_agent, gamma, **kwargs):
    """Get BAPOSGMCP init function.

    This function which handles dynamic loading of other agent policies.
    This is needed to ensure independent policies are used for each experiment
    when running experiments in parallel.
    """
    args = kwargs.pop("args")

    other_agent_id = (ego_agent + 1) % 2
    other_policies = {
        other_agent_id: _load_agent_policies(args, other_agent_id)
    }

    if args.rollout_policy_id is None:
        rollout_policy = ba_policy_lib.RandomPolicy(
            model, ego_agent, gamma
        )
    else:
        rollout_policy = _load_agent_policy(
            args, args.rollout_policy_id, ego_agent
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


def _create_exp_params(args):
    print("== Creating Experiments ==")
    sample_env = get_base_env(args)
    env_model = sample_env.model

    # env is symmetric so only need to run BAPOSGMCP for a single agent
    baposgmcp_agent_id = 0

    exp_params_list = []
    exp_id = 0
    for num_sims in args.num_sims:
        baposgmcp_params = exp_lib.PolicyParams(
            name=f"BAPOSGMCP_{baposgmcp_agent_id}",
            gamma=args.gamma,
            kwargs={
                "args": args,          # is removed by custom init fn
                "other_policy_prior": None,     # uniform
                "num_sims": num_sims,
                "c_init": 1.0,
                "c_base": 100.0,
                "truncated": True,
                "reinvigorator": tree_lib.BABeliefRejectionSampler(env_model),
                "extra_particles_prop": 1.0 / 16,
                "step_limit": sample_env.spec.max_episode_steps,
                "epsilon": 0.01
            },
            init=_baposgmcp_init_fn
        )

        other_agent_policy_params = _load_agent_policy_params(args)
        for policy_params in other_agent_policy_params:
            policies = [baposgmcp_params, policy_params]

            exp_params = exp_lib.ExpParams(
                exp_id=exp_id,
                env_name=args.env_name,
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

            if args.debug and exp_id == args.n_procs:
                break
        if args.debug and exp_id == args.n_procs:
            break

    return exp_params_list


def _main(args):
    print(f"== Running Experiment for env {args.env_name} ==")
    # check env name is valid
    get_base_env(args)

    register_env(args.env_name, registered_env_creator)

    exp_params_list = _create_exp_params(args)

    # Experiments: Run BAPOSGMCP against each RLLib policy for both agents
    result_dir = osp.join(EXP_RESULTS_DIR, str(datetime.now()))
    pathlib.Path(result_dir).mkdir(exist_ok=False)

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

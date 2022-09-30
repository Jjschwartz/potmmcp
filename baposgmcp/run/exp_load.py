import copy
import argparse
from itertools import product
from typing import List, Optional, Dict, Sequence

import posggym
import posggym_agents

from baposgmcp.run.exp import ExpParams, PolicyParams
from baposgmcp.run.render import Renderer, EpisodeRenderer


def get_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default experiment args."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--init_seed", type=int, default=0,
        help="Experiment start seed."
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1,
        help="Number of seeds to use."
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
        "--run_exp_id", type=int, default=None,
        help="Run only exp with specific ID. If None will run all exps."
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
        "--record_env", action="store_true",
        help="Record renderings of experiment episodes."
    )
    parser.add_argument(
        "--root_save_dir", type=str, default=None,
        help=(
            "Optional directory to save results in. If supplied then it must "
            "be an existing directory. If None uses default "
            "~/baposgmcp_results/<env_id>/ dir as root results dir."
        )
    )
    return parser


def env_renderer_fn() -> Sequence[Renderer]:
    """Get environment renderer."""
    return [EpisodeRenderer()]


def get_pairwise_exp_params(env_id: str,
                            policy_params: List[List[PolicyParams]],
                            init_seed: int,
                            num_seeds: int,
                            num_episodes: int,
                            discount: float,
                            time_limit: Optional[int] = None,
                            exp_id_init: int = 0,
                            tracker_fn: Optional = None,
                            tracker_fn_kwargs: Optional[Dict] = None,
                            renderer_fn: Optional = None,
                            record_env: bool = False,
                            **kwargs) -> List[ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Will create an experiment for every possible pairing of policies.
    """
    assert isinstance(policy_params[0], list)
    env = posggym.make(env_id)
    episode_step_limit = env.spec.max_episode_steps

    exp_params_list = []
    for i, (exp_seed, *policies) in enumerate(product(
            range(num_seeds), *policy_params
    )):
        policies = [*policies]

        exp_params = ExpParams(
            exp_id=exp_id_init+i,
            env_id=env_id,
            policy_params_list=policies,
            discount=discount,
            seed=init_seed + exp_seed,
            num_episodes=num_episodes,
            episode_step_limit=episode_step_limit,
            time_limit=time_limit,
            tracker_fn=tracker_fn,
            tracker_fn_kwargs=tracker_fn_kwargs,
            renderer_fn=renderer_fn,
            record_env=record_env,
            record_env_freq=max(1, num_episodes // 10),
            use_checkpointing=True,
        )
        exp_params_list.append(exp_params)

    return exp_params_list


def posggym_agent_entry_point(model, agent_id, kwargs):
    """Initialize a posggym agent.

    Required kwargs
    ---------------
    policy_id: str
    """
    kwargs = copy.deepcopy(kwargs)
    policy_id = kwargs.pop("policy_id")
    return posggym_agents.make(policy_id, model, agent_id, **kwargs)


def load_posggym_agent_params(policy_ids: List[str]) -> List[PolicyParams]:
    """Load posggym-agent policy params from ids."""
    return [
        PolicyParams(
            id=policy_id,
            entry_point=posggym_agent_entry_point,
            kwargs={"policy_id": policy_id},
            info=None
        )
        for policy_id in policy_ids
    ]

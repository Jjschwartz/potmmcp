"""Functions for running Pairwise comparison experiments for policies."""
import argparse
from typing import List, Sequence, Optional
from itertools import product, combinations_with_replacement

import posggym

from posggym_agents.exp.render import Renderer, EpisodeRenderer
from posggym_agents.exp.exp import ExpParams, get_exp_parser


def _renderer_fn() -> Sequence[Renderer]:
    return [EpisodeRenderer()]


def get_symmetric_pairwise_exp_parser() -> argparse.ArgumentParser:
    """Get arg parser with default symmetric env pairwise experiment args.

    Inherits arguments from the posgyym_agent.exp.get_exp_parser() parser.
    """
    parser = get_exp_parser()
    parser.add_argument(
        "--env_id", type=str,
        help="Name of the environment to run experiment in."
    )
    parser.add_argument(
        "-pids", "--policy_ids", type=str, nargs="+", default=None,
        help=(
            "List of IDs of policies to compare, if None will run all policies"
            " available for the given environment."
        )
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
        "--render", action="store_true",
        help="Render experiment episodes."
    )
    parser.add_argument(
        "--record_env", action="store_true",
        help="Record renderings of experiment episodes."
    )
    return parser


def get_symmetric_pairwise_exp_params(env_id: str,
                                      policy_ids: Optional[Sequence[str]],
                                      init_seed: int,
                                      num_seeds: int,
                                      num_episodes: int,
                                      time_limit: Optional[int] = None,
                                      exp_id_init: int = 0,
                                      render: bool = False,
                                      record_env: bool = True,
                                      **kwargs) -> List[ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Assumes that the environment is symmetric.
    - Will create an experiment for every possible pairing of policy ids.
    """
    env = posggym.make(env_id)
    assert env.is_symmetric

    if policy_ids is None:
        from posggym_agents.agents.registration import registry
        policy_ids = [spec.id for spec in registry.all_for_env(env_id, True)]

    exp_params_list = []
    for i, (exp_seed, policies) in enumerate(product(
        range(num_seeds),
        combinations_with_replacement(policy_ids, env.n_agents)
    )):
        exp_params = ExpParams(
            exp_id=exp_id_init+i,
            env_id=env_id,
            policy_ids=policies,
            seed=init_seed+exp_seed,
            num_episodes=num_episodes,
            time_limit=time_limit,
            tracker_fn=None,
            renderer_fn=_renderer_fn if render else None,
            record_env=record_env,
            record_env_freq=None
        )
        exp_params_list.append(exp_params)

    return exp_params_list


def get_asymmetric_pairwise_exp_parser() -> argparse.ArgumentParser:
    """Get arg parser with default asymmetric env pairwise experiment args.

    Inherits arguments from the posgyym_agent.exp.get_exp_parser() parser.
    """
    parser = get_exp_parser()
    parser.add_argument(
        "--env_id", type=str,
        help="Name of the environment to run experiment in."
    )
    parser.add_argument(
        "-pids", "--policy_ids",
        type=str, nargs="+", action="append", default=None,
        help=(
            "List of IDs of policies to compare for each agent. If specified, "
            "Then this flag must be used (followed by list of policy IDs for "
            "the agent) once for each agent in the environment. "
            "E.g. for env with 2 agents and 2 policies for each agent you "
            "would use: `-pids pi_0_0 pi_1_0 -pids pi_0_1 pi_1_1`. "
            "If it is not used (i.e. is None) then all policies available for "
            "the given environment will be compared."
        )
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
        "--render", action="store_true",
        help="Render experiment episodes."
    )
    parser.add_argument(
        "--record_env", action="store_true",
        help="Record renderings of experiment episodes."
    )
    return parser


def get_asymmetric_pairwise_exp_params(env_id: str,
                                       policy_ids: Optional[
                                           Sequence[Sequence[str]]
                                       ],
                                       init_seed: int,
                                       num_seeds: int,
                                       num_episodes: int,
                                       time_limit: Optional[int] = None,
                                       exp_id_init: int = 0,
                                       render: bool = False,
                                       record_env: bool = True,
                                       **kwargs) -> List[ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Assumes that the environment is not symmetric.
    - Requires a list of policy_ids for each agent in the environment
    - Will create an experiment for every possible pairing of policy ids.
    """
    env = posggym.make(env_id)
    assert not env.is_symmetric

    if policy_ids is None:
        from posggym_agents.agents.registration import registry
        policy_ids = [[] for _ in range(env.n_agents)]
        policy_specs = registry.all_for_env(env_id, True)
        for spec in policy_specs:
            valid_agent_ids = spec.valid_agent_ids
            if not valid_agent_ids:
                # valied for all agents
                valid_agent_ids = range(env.n_agents)
            for i in valid_agent_ids:
                policy_ids[i].append(spec.id)

    assert len(policy_ids) == env.n_agents
    assert all(len(p) > 0 for p in policy_ids)

    exp_params_list = []
    for i, (exp_seed, policies) in enumerate(product(
            range(num_seeds), product(*policy_ids)
    )):
        exp_params = ExpParams(
            exp_id=exp_id_init+i,
            env_id=env_id,
            policy_ids=policies,
            seed=init_seed+exp_seed,
            num_episodes=num_episodes,
            time_limit=time_limit,
            tracker_fn=None,
            renderer_fn=_renderer_fn if render else None,
            record_env=record_env,
            record_env_freq=None
        )
        exp_params_list.append(exp_params)

    return exp_params_list

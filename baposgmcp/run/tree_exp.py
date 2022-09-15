import copy
import argparse
from itertools import product
from typing import List, Sequence, Optional, Dict, Callable

import posggym
import posggym.model as M

import baposgmcp.policy as P
import baposgmcp.tree as tree_lib
from baposgmcp.meta_policy import DictMetaPolicy
from baposgmcp.policy_prior import MapPolicyPrior

import baposgmcp.run.stats as stats_lib
from baposgmcp.run.render import Renderer, EpisodeRenderer
from baposgmcp.run.exp import ExpParams, PolicyParams, get_exp_parser


def get_baposgmcp_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default baposgmc experiment args.

    Inherits arguments from the baposgmcp.run.exp.get_exp_parser() parser.
    """
    parser = get_exp_parser()
    parser.add_argument(
        "--env_name", type=str, required=True,
        help="Name of the environment to run experiment in."
    )
    parser.add_argument(
        "--baposgmcp_policy_ids", type=str, nargs="+", required=True,
        help=(
            "IDs of policies for BAPOSGMCP to "
            "use in it's other agent prior and meta policy."
        )
    )
    parser.add_argument(
        "--other_policy_ids", type=str, nargs="+", default=None,
        help=(
            "IDs of policies to test against. "
            "If None uses values from 'baposgmcp_policy_ids'."
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
        "--discount", type=float, default=0.99,
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
        "--num_sims", type=int, nargs="*", default=[128],
        help="Number of simulations per search."
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


def baposgmcp_init_fn(model: M.POSGModel, agent_id: M.AgentID, kwargs):
    """Initialize BAPOSGMCP policy.

    This function which handles dynamic loading of other agent and
    rollout policies. This is required to ensure independent policies are used
    for each experiment when running experiments in parallel.

    Required kwargs
    ---------------
    other_policy_prior : P.AgentPolicyDist
    meta_policy_dict : Dict[P.PolicyState, P.PolicyDist]

    """
    # need to do copy as kwargs is modified
    # and may be reused in a different experiment if done on the same CPU
    kwargs = copy.deepcopy(kwargs)

    other_policy_prior = MapPolicyPrior.load_posggym_agents_prior(
        model,
        agent_id,
        policy_dist_map=kwargs.pop("other_policy_dist")
    )

    meta_policy = DictMetaPolicy.load_possgym_agents_meta_policy(
        model,
        agent_id,
        meta_policy_dict=kwargs.pop("meta_policy_dict")
    )

    return tree_lib.BAPOSGMCP(
        model,
        agent_id,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        **kwargs
    )


def load_baposgmcp_params(env_name: str,
                          agent_id: M.AgentID,
                          discount: float,
                          num_sims: List[int],
                          baposgmcp_kwargs: Dict,
                          other_policy_dist: P.AgentPolicyDist,
                          meta_policy_dict: Dict[P.PolicyState, P.PolicyDist]
                          ) -> List[PolicyParams]:
    """Load list of policy params for BAPOSGMCP policy."""
    base_kwargs = dict(baposgmcp_kwargs)
    env = posggym.make(env_name)

    base_kwargs.update({
        "discount": discount,
        "other_policy_dist": other_policy_dist,
        "meta_policy_dict": meta_policy_dict
    })

    if "policy_id" not in base_kwargs:
        base_kwargs["policy_id"] = "baposgmcp"

    policy_params = []
    for n in num_sims:
        # need to do copy as kwargs is modified in baposgmcp init fn
        kwargs = copy.deepcopy(base_kwargs)
        kwargs["reinvigorator"] = tree_lib.BABeliefRejectionSampler(env.model)
        kwargs["num_sims"] = n
        kwargs["policy_id"] = f"{kwargs['policy_id']}_{n}"

        baposgmcp_params = PolicyParams(
            id=f"BAPOSGMCP_{n}",
            kwargs=kwargs,
            entry_point=baposgmcp_init_fn,
            info={}
        )
        policy_params.append(baposgmcp_params)

    return policy_params


def _renderer_fn() -> Sequence[Renderer]:
    return [
        EpisodeRenderer(),
        # run_lib.SearchTreeRenderer(2)
    ]


def get_baposgmcp_tracker_fn(num_agents: int,
                             discount: float,
                             step_limit: int
                             ) -> Callable[[], Sequence[stats_lib.Tracker]]:
    """Get tracker generator function for BAPOSGMCP experiment."""
    def tracker_fn():
        trackers = stats_lib.get_default_trackers(num_agents, discount)

        # disbling for now while I test some things
        # tracker_kwargs = {
        #     "num_agents": num_agents,
        #     "track_per_step": True,
        #     "step_limit": step_limit
        # }
        # trackers.append(stats_lib.BayesAccuracyTracker(**tracker_kwargs))
        # trackers.append(
        #     stats_lib.ActionDistributionDistanceTracker(**tracker_kwargs)
        # )
        # trackers.append(
        #     stats_lib.BeliefHistoryAccuracyTracker(**tracker_kwargs)
        # )
        # trackers.append(
        #     stats_lib.BeliefStateAccuracyTracker(**tracker_kwargs)
        # )
        return trackers
    return tracker_fn


def get_baposgmcp_exp_params(env_name: str,
                             baposgmcp_params: List[PolicyParams],
                             other_policy_params: List[List[PolicyParams]],
                             init_seed: int,
                             num_seeds: int,
                             num_episodes: int,
                             discount: float,
                             time_limit: Optional[int] = None,
                             exp_id_init: int = 0,
                             render: bool = False,
                             record_env: bool = True,
                             baposgmcp_agent_id: M.AgentID = 0,
                             **kwargs) -> List[ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Assumes that the environment is symmetric.
    - Will create an experiment for every possible pairing of policies.
    """
    assert isinstance(other_policy_params[0], list)
    env = posggym.make(env_name)
    episode_step_limit = env.spec.max_episode_steps

    exp_params_list = []
    for i, (exp_seed, baposgmcp_policy, *other_policies) in enumerate(product(
            range(num_seeds),
            baposgmcp_params,
            *other_policy_params,
    )):
        policies = [*other_policies]
        policies.insert(baposgmcp_agent_id, baposgmcp_policy)

        exp_params = ExpParams(
            exp_id=exp_id_init+i,
            env_name=env_name,
            policy_params_list=policies,
            discount=discount,
            seed=init_seed + exp_seed,
            num_episodes=num_episodes,
            episode_step_limit=episode_step_limit,
            time_limit=time_limit,
            tracker_fn=get_baposgmcp_tracker_fn(
                env.n_agents, discount, episode_step_limit
            ),
            renderer_fn=_renderer_fn if render else None,
            record_env=record_env,
            record_env_freq=max(1, num_episodes // 10),
            use_checkpointing=True,
        )
        exp_params_list.append(exp_params)

    return exp_params_list

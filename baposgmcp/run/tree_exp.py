import copy
import argparse
from itertools import product
from typing import List, Sequence, Optional, Dict

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

    Plus any other arguments required by BAPOSMGPCP.__init__ (excluding
    for model, agent_id, other_policy_prior and meta_policy)

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

    if "reinvigorator" not in kwargs:
        kwargs["reinvigorator"] = tree_lib.BABeliefRejectionSampler(model)

    return tree_lib.BAPOSGMCP(
        model,
        agent_id,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        **kwargs
    )


def load_baposgmcp_params(num_sims: List[int],
                          baposgmcp_kwargs: Dict,
                          other_policy_dist: P.AgentPolicyDist,
                          meta_policy_dict: Dict[P.PolicyState, P.PolicyDist]
                          ) -> List[PolicyParams]:
    """Load list of policy params for BAPOSGMCP with different num sims."""
    # copy
    base_kwargs = dict(baposgmcp_kwargs)

    base_kwargs.update({
        "other_policy_dist": other_policy_dist,
        "meta_policy_dict": meta_policy_dict
    })

    if "policy_id" not in base_kwargs:
        base_kwargs["policy_id"] = "baposgmcp"

    policy_params = []
    for n in num_sims:
        # need to do copy as kwargs is modified in baposgmcp init fn
        kwargs = copy.deepcopy(base_kwargs)
        policy_id = f"{kwargs['policy_id']}_{n}"
        kwargs["num_sims"] = n
        kwargs["policy_id"] = policy_id

        baposgmcp_params = PolicyParams(
            id=policy_id,
            kwargs=kwargs,
            entry_point=baposgmcp_init_fn,
            info={}
        )
        policy_params.append(baposgmcp_params)

    return policy_params


def _renderer_fn() -> Sequence[Renderer]:
    return [EpisodeRenderer()]


def _renderer_tree_fn() -> Sequence[Renderer]:
    from baposgmcp.run.render import SearchTreeRenderer
    return [
        EpisodeRenderer(),
        SearchTreeRenderer(1)
    ]


def baposgmcp_tracker_fn(kwargs) -> Sequence[stats_lib.Tracker]:
    """Get trackers for BAPOSGMCP experiment.

    Required kwargs
    ---------------
    num_agents : int
    discount : float
    step_limit : int

    """
    num_agents = kwargs["num_agents"]
    discount = kwargs["discount"]

    trackers = stats_lib.get_default_trackers(num_agents, discount)

    tracker_kwargs = {
        "num_agents": num_agents,
        # only track per step if step limit is provided
        "track_per_step": kwargs["step_limit"] is not None,
        "step_limit": kwargs["step_limit"]
    }
    trackers.append(stats_lib.BayesAccuracyTracker(**tracker_kwargs))
    trackers.append(
        stats_lib.ActionDistributionDistanceTracker(**tracker_kwargs)
    )
    # trackers.append(
    #     stats_lib.BeliefHistoryAccuracyTracker(**tracker_kwargs)
    # )
    # trackers.append(
    #     stats_lib.BeliefStateAccuracyTracker(**tracker_kwargs)
    # )
    return trackers


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
                             render_tree: bool = False,
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

    if not render:
        renderer_fn = None
    elif render_tree:
        renderer_fn = _renderer_tree_fn
    else:
        renderer_fn = _renderer_fn

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
            tracker_fn=baposgmcp_tracker_fn,
            tracker_fn_kwargs={
                "num_agents": env.n_agents,
                "discount": discount,
                "step_limit": episode_step_limit
            },
            renderer_fn=renderer_fn,
            record_env=record_env,
            record_env_freq=max(1, num_episodes // 10),
            use_checkpointing=True,
        )
        exp_params_list.append(exp_params)

    return exp_params_list

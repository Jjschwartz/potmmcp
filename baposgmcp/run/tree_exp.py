import copy
import argparse
from itertools import product
from typing import List, Sequence, Optional, Dict

import posggym
import posggym.model as M

from baposgmcp import pbt
import baposgmcp.policy as P
import baposgmcp.tree as tree_lib
from baposgmcp.meta_policy import MetaPolicy, SingleMetaPolicy, DictMetaPolicy
from baposgmcp.policy_prior import (
    PolicyPrior, UniformPolicyPrior, MapPolicyPrior
)

import baposgmcp.run.stats as stats_lib
from baposgmcp.run.runner import RunConfig
from baposgmcp.run.rl_exp import load_rllib_agent_policy
from baposgmcp.run.render import Renderer, EpisodeRenderer
from baposgmcp.run.exp import ExpParams, PolicyParams, get_exp_parser

# TODO Update load_agent_policies to use policy_id and posggym-agents


def get_baposgmcp_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default baposgmc experiment args.

    Inherits argumenrts from the baposgmcp.run.exp.get_exp_parser() parser.
    """
    parser = get_exp_parser()
    parser.add_argument(
        "--env_name", type=str, required=True,
        help="Name of the environment to run experiment in."
    )
    parser.add_argument(
        "--baposgmcp_policy_dirs", type=str, nargs="+", required=True,
        help=(
            "Paths to dirs containing trained RL policies for BAPOSGMCP to "
            "use in it's other agent prior and meta policy."
        )
    )
    parser.add_argument(
        "--other_policy_dirs", type=str, nargs="+", default=None,
        help=(
            "Paths to dirs containing trained RL policies to test against. "
            "If None uses values from 'baposgmcp_policy_dirs'."
        )
    )
    parser.add_argument(
        "--baposgmcp_policy_ids", type=str, default=None, nargs="*",
        help=(
            "ID/s of policy to use in other agent prior for BAPOSGMCP, if None"
            " then uses all the policies 'baposgmcp_policy_dirs'."
        )
    )
    parser.add_argument(
        "--other_policy_ids", type=str, default=None, nargs="*",
        help=(
            "ID/s of policy to use for other agent in experiments, if None "
            "then uses value from 'baposgmcp_policy_ids'."
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
        "--gamma", type=float, default=0.99,
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


def load_agent_policies(env_name: str,
                        policy_dir: str,
                        ego_agent: M.AgentID,
                        gamma: float,
                        include_random_policy: bool = True,
                        include_policy_ids: Optional[List[str]] = None,
                        policy_load_kwargs: Optional[Dict] = None,
                        ) -> Dict[str, P.BasePolicy]:
    """Load agent rllib policies from file.

    include_policy_ids is an optional list of policy IDs specifying which
    policies to import. If it is None then all policies in the policy dir are
    imported.

    See 'load_rllib_agent_policy' for info on policy_load_kwargs. Note that
    'policy_id', 'policy_dir' arguments are populated in this function so
    shouldn't be included.

    """
    env = posggym.make(env_name)

    igraph = ba_rllib.import_igraph(policy_dir, True)

    if policy_load_kwargs is None:
        policy_load_kwargs = {}

    policies: Dict[str, P.BasePolicy] = {}
    random_policy_added = False
    for policy_id in igraph.policies[pbt.InteractionGraph.SYMMETRIC_ID]:
        if (
            include_policy_ids is not None
            and policy_id not in include_policy_ids
        ):
            continue

        policy_id = str(policy_id)
        if "-1" in policy_id:
            policies[policy_id] = P.RandomPolicy(
                env.model, ego_agent, gamma, policy_id=policy_id
            )
            random_policy_added = True
        else:
            kwargs = dict(policy_load_kwargs)
            kwargs.update({
                "env_name": env_name,
                "policy_dir": policy_dir,
                "policy_id": policy_id,
            })
            policies[policy_id] = load_rllib_agent_policy(
                env.model, ego_agent, gamma, **kwargs
            )

    if include_random_policy and not random_policy_added:
        policies["pi_-1"] = P.RandomPolicy(
            env.model, ego_agent, gamma, policy_id="pi_-1"
        )

    return policies


def _renderer_fn(kwargs) -> Sequence[Renderer]:
    if kwargs["render"]:
        return [
            EpisodeRenderer(),
            # run_lib.SearchTreeRenderer(2)
        ]
    return []


def _tracker_fn(policies: List[P.BasePolicy],
                kwargs) -> List[stats_lib.Tracker]:
    trackers = stats_lib.get_default_trackers(policies)

    tracker_kwargs = {
        "num_agents": len(policies),
        "track_per_step": True,
        "step_limit": kwargs["step_limit"]
    }

    trackers.append(stats_lib.BayesAccuracyTracker(**tracker_kwargs))
    trackers.append(
        stats_lib.ActionDistributionDistanceTracker(**tracker_kwargs)
    )
    trackers.append(stats_lib.BeliefHistoryAccuracyTracker(**tracker_kwargs))
    trackers.append(stats_lib.BeliefStateAccuracyTracker(**tracker_kwargs))
    return trackers


def _load_other_policy_prior(env_name: str,
                             model: M.POSGModel,
                             ego_agent: M.AgentID,
                             gamma: float,
                             other_policy_dir: str,
                             other_policy_ids: Optional[List[str]],
                             other_policy_dist: Optional[P.AgentPolicyDist]
                             ) -> PolicyPrior:
    other_policies = {}
    for i in range(model.n_agents):
        if i == ego_agent:
            continue
        other_policies[i] = load_agent_policies(
            env_name,
            other_policy_dir,
            i,
            gamma,
            include_random_policy=False,
            include_policy_ids=other_policy_ids,
            # use defaults in rl_exp.load_rllib_agent_policy
            policy_load_kwargs=None
        )

    if other_policy_dist is None:
        other_policy_prior = UniformPolicyPrior(
            model, ego_agent, other_policies
        )
    else:
        other_policy_prior = MapPolicyPrior(
            model, ego_agent, other_policies, other_policy_dist
        )
    return other_policy_prior


def _load_meta_policy(env_name: str,
                      model: M.POSGModel,
                      ego_agent: M.AgentID,
                      gamma: float,
                      meta_policy_dir: Optional[str],
                      meta_policy_dict: Optional[
                          Dict[P.PolicyState, P.PolicyDist]
                      ]) -> MetaPolicy:
    if meta_policy_dir is None:
        ego_policies = {
            "pi_-1": P.RandomPolicy(model, ego_agent, gamma, policy_id="pi_-1")
        }
        meta_policy = SingleMetaPolicy(model, ego_agent, ego_policies)
    else:
        assert meta_policy_dict is not None

        meta_policy_ids = set()
        for pi_dist in meta_policy_dict.values():
            meta_policy_ids.update(pi_dist)

        ego_policies = load_agent_policies(
            env_name,
            meta_policy_dir,
            ego_agent,
            gamma,
            include_random_policy=False,
            include_policy_ids=list(meta_policy_ids),
            # use defaults in rl_exp.load_rllib_agent_policy
            policy_load_kwargs=None
        )
        meta_policy = DictMetaPolicy(
            model, ego_agent, ego_policies, meta_policy_dict
        )
    return meta_policy


def baposgmcp_init_fn(model: M.POSGModel,
                      ego_agent: M.AgentID,
                      gamma: float,
                      **kwargs):
    """Get BAPOSGMCP init function.

    This function which handles dynamic loading of other agent and
    rollout policies. This is required to ensure independent policies are used
    for each experiment when running experiments in parallel.

    Required kwargs
    ---------------
    env_name : str
    other_policy_dir : str

    Optional kwargs (defaults)
    --------------------------
    other_policy_prior : P.AgentPolicyDist (None = uniform prior)
    other_policy_ids : List[str] (None = all policies in policy dir)
    meta_policy_dir : str (None = Random rollout)
    meta_policy_dict : Dict[P.PolicyState, P.PolicyDist] (None = Random only)

    Note if meta_policy_dir is defined then meta_policy_dict must also be
    defined.

    """
    env_name = kwargs.pop("env_name")

    other_policy_prior = _load_other_policy_prior(
        env_name,
        model,
        ego_agent,
        gamma,
        other_policy_dir=kwargs.pop("other_policy_dir"),
        other_policy_ids=kwargs.pop("other_policy_ids", None),
        other_policy_dist=kwargs.pop("other_policy_prior", None)
    )

    meta_policy = _load_meta_policy(
        env_name,
        model,
        ego_agent,
        gamma,
        meta_policy_dir=kwargs.pop("meta_policy_dir", None),
        meta_policy_dict=kwargs.pop("meta_policy_dict", None)
    )

    return tree_lib.BAPOSGMCP(
        model,
        ego_agent,
        gamma,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        **kwargs
    )


def load_baposgmcp_params(env_name: str,
                          ego_agent: M.AgentID,
                          gamma: float,
                          num_sims: List[int],
                          baposgmcp_kwargs: Dict,
                          other_policy_dir: str,
                          other_policy_ids: Optional[List[str]],
                          other_policy_dist: Optional[P.AgentPolicyDist],
                          meta_policy_dir: Optional[str],
                          meta_policy_dict: Optional[
                              Dict[P.PolicyState, P.PolicyDist]
                          ]) -> List[PolicyParams]:
    """Load list of policy params for BAPOSGMCP policy."""
    base_kwargs = dict(baposgmcp_kwargs)
    env = posggym.make(env_name)

    base_kwargs.update({
        "env_name": env_name,
        "policy_id": "pi_baposgmcp",
        "other_policy_dir": other_policy_dir,
        "other_policy_ids": other_policy_ids,
        "other_policy_dist": other_policy_dist,
        "meta_policy_dir": meta_policy_dir,
        "meta_policy_dict": meta_policy_dict
    })

    # for reporting in results
    info = {
        "other_policy_dir": other_policy_dir,
        "meta_policy_dir": meta_policy_dir,
    }

    policy_params = []
    for n in num_sims:
        # need to do copy as kwargs is modified in baposgmcp init fn
        kwargs = copy.deepcopy(base_kwargs)
        kwargs["reinvigorator"] = tree_lib.BABeliefRejectionSampler(env.model)
        kwargs["num_sims"] = n

        baposgmcp_params = PolicyParams(
            name=f"BAPOSGMCP_{ego_agent}",
            gamma=gamma,
            kwargs=kwargs,
            init=baposgmcp_init_fn,
            info=info
        )
        policy_params.append(baposgmcp_params)

    return policy_params


def get_baposgmcp_exp_params(env_name: str,
                             baposgmcp_params: List[PolicyParams],
                             other_policy_params: List[List[PolicyParams]],
                             init_seed: int,
                             num_seeds: int,
                             num_episodes: int,
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
    env = posggym.make(env_name)
    episode_step_limit = env.spec.max_episode_steps

    exp_params_list = []
    for i, (exp_seed, baposgmcp_policy, other_policies) in enumerate(product(
            range(num_seeds),
            baposgmcp_params,
            other_policy_params,
    )):
        policies = [*other_policies]
        policies.insert(baposgmcp_agent_id, baposgmcp_policy)

        exp_params = ExpParams(
            exp_id=exp_id_init+i,
            env_name=env_name,
            policy_params_list=policies,
            run_config=RunConfig(
                seed=init_seed + exp_seed,
                num_episodes=num_episodes,
                episode_step_limit=episode_step_limit,
                time_limit=time_limit,
                use_checkpointing=True
            ),
            tracker_fn=_tracker_fn,
            tracker_kwargs={"step_limit": episode_step_limit},
            renderer_fn=_renderer_fn,
            renderer_kwargs={"render": render},
            record_env=record_env,
            record_env_freq=max(1, num_episodes // 10)
        )
        exp_params_list.append(exp_params)

    return exp_params_list

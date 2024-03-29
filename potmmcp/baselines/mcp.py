"""Baselines of POTMMCP using different meta-policies."""
import copy
import warnings
from itertools import product
from typing import Dict, List, Union

import potmmcp.policy as P
import potmmcp.tree as tree_lib
import posggym.model as M
from potmmcp.meta_policy import SingleMetaPolicy
from potmmcp.policy_prior import load_posggym_agents_policy_prior
from potmmcp.run.exp import PolicyParams


def fixed_pi_potmmcp_entry_point(model: M.POSGModel, agent_id: M.AgentID, kwargs):
    """Initialize POTMMCP with a single rollout policy.

    This function which handles dynamic loading of other agent prior and
    the random rollout policy. This is required to ensure independent policies
    are used for each experiment when running experiments in parallel.

    Required kwargs
    ---------------
    policy_prior_map: Union[P.AgentPolicyDist, Dict[P.PolicyState, float]]
    fixed_policy_id : str

    Plus any other arguments required by POTMMCP.__init__ (excluding
    for model, agent_id, other_policy_prior and meta_policy)

    """
    # need to do copy as kwargs is modified
    # and may be reused in a different experiment if done on the same CPU
    kwargs = copy.deepcopy(kwargs)

    other_policy_prior = load_posggym_agents_policy_prior(
        model, agent_id, policy_prior_map=kwargs.pop("policy_prior_map")
    )

    meta_policy = SingleMetaPolicy.load_possgym_agents_meta_policy(
        model, agent_id, policy_id=kwargs.pop("fixed_policy_id")
    )

    return tree_lib.POTMMCP(
        model,
        agent_id,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        **kwargs,
    )


def load_random_potmmcp_params(
    variable_params: Dict[str, List],
    potmmcp_kwargs: Dict,
    policy_prior_map: Union[P.AgentPolicyDist, Dict[P.PolicyState, float]],
    base_policy_id: str = "mcp-random",
) -> List[PolicyParams]:
    """Load params for POTMMCP using random policy for evaluation.

    This is POTMMCP using a meta-policy which always returns the uniform
    random policy.
    """
    base_kwargs = copy.deepcopy(potmmcp_kwargs)

    if "truncated" not in base_kwargs or base_kwargs["truncated"]:
        warnings.warn(
            "Cannot do truncated search with random policy. Changing to "
            "untruncated search."
        )
        base_kwargs["truncated"] = False

    base_kwargs.update(
        {"policy_prior_map": policy_prior_map, "fixed_policy_id": "random-v0"}
    )

    policy_params = []
    param_names = list(variable_params)
    param_value_lists = [variable_params[k] for k in param_names]
    for values in product(*param_value_lists):
        # need to do copy as kwargs is modified in potmmcp init fn
        kwargs = copy.deepcopy(base_kwargs)
        policy_id = base_policy_id
        for k, v in zip(param_names, values):
            kwargs[k] = v
            # remove _ so it's easier to parse when doing analysis
            policy_id += f"_{k.replace('_', '')}{v}"
        kwargs["policy_id"] = policy_id

        potmmcp_params = PolicyParams(
            id=policy_id,
            kwargs=kwargs,
            entry_point=fixed_pi_potmmcp_entry_point,
            info={},
        )
        policy_params.append(potmmcp_params)

    return policy_params


def load_fixed_pi_potmmcp_params(
    variable_params: Dict[str, List],
    fixed_policy_ids: List[str],
    potmmcp_kwargs: Dict,
    policy_prior_map: Union[P.AgentPolicyDist, Dict[P.PolicyState, float]],
    base_policy_id: str = "mcp-fixed",
) -> List[PolicyParams]:
    """Load params for POTMMCP using fixed policy for evaluation.

    This is POTMMCP using a meta-policy which always returns the same
    fixed policy.
    """
    base_kwargs = copy.deepcopy(potmmcp_kwargs)
    base_kwargs.update(
        {
            "policy_prior_map": policy_prior_map,
        }
    )

    policy_params = []
    param_names = list(variable_params)
    param_value_lists = [variable_params[k] for k in param_names]
    for pi_id, values in product(fixed_policy_ids, product(*param_value_lists)):
        # need to do copy as kwargs is modified in potmmcp init fn
        kwargs = copy.deepcopy(base_kwargs)
        kwargs["fixed_policy_id"] = pi_id
        short_pi_id = pi_id.split("/")[-1].replace("_", "")
        policy_id = f"{base_policy_id}_pi{short_pi_id}"
        for k, v in zip(param_names, values):
            kwargs[k] = v
            # remove _ so it's easier to parse when doing analysis
            policy_id += f"_{k.replace('_', '')}{v}"
        kwargs["policy_id"] = policy_id

        potmmcp_params = PolicyParams(
            id=policy_id,
            kwargs=kwargs,
            entry_point=fixed_pi_potmmcp_entry_point,
            info={},
        )
        policy_params.append(potmmcp_params)

    return policy_params

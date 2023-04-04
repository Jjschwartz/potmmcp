import copy
from itertools import product
from typing import Dict, List, Sequence, Union

import posggym.model as M

import potmmcp.policy as P
import potmmcp.tree as tree_lib
from potmmcp.meta_policy import DictMetaPolicy
from potmmcp.policy_prior import load_posggym_agents_policy_prior
from potmmcp.run.exp import PolicyParams
from potmmcp.run.render import EpisodeRenderer, Renderer


def potmmcp_init_fn(model: M.POSGModel, agent_id: M.AgentID, kwargs):
    """Initialize POTMMCP policy.

    This function which handles dynamic loading of other agent and
    rollout policies. This is required to ensure independent policies are used
    for each experiment when running experiments in parallel.

    Required kwargs
    ---------------
    policy_prior_map: Union[P.AgentPolicyDist, Dict[P.PolicyState, float]]
    meta_policy_dict : Dict[P.PolicyState, P.PolicyDist]

    Plus any other arguments required by BAPOSMGPCP.__init__ (excluding
    for model, agent_id, other_policy_prior and meta_policy)

    """
    # need to do copy as kwargs is modified
    # and may be reused in a different experiment if done on the same CPU
    kwargs = copy.deepcopy(kwargs)

    other_policy_prior = load_posggym_agents_policy_prior(
        model, agent_id, policy_prior_map=kwargs.pop("policy_prior_map")
    )

    meta_policy = DictMetaPolicy.load_possgym_agents_meta_policy(
        model, agent_id, meta_policy_dict=kwargs.pop("meta_policy_dict")
    )

    return tree_lib.POTMMCP(
        model,
        agent_id,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        **kwargs,
    )


def load_potmmcp_params(
    variable_params: Dict[str, List],
    potmmcp_kwargs: Dict,
    policy_prior_map: Union[P.AgentPolicyDist, Dict[P.PolicyState, float]],
    meta_policy_dict: Dict[P.PolicyState, P.PolicyDist],
    base_policy_id: str = "potmmcp",
) -> List[PolicyParams]:
    """Load list of policy params for POTMMCP with different num sims."""
    base_kwargs = copy.deepcopy(potmmcp_kwargs)

    base_kwargs.update(
        {"policy_prior_map": policy_prior_map, "meta_policy_dict": meta_policy_dict}
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
            if isinstance(v, float):
                policy_id += f"_{k.replace('_', '')}{v:.4f}"
            else:
                policy_id += f"_{k.replace('_', '')}{v}"
        kwargs["policy_id"] = policy_id

        potmmcp_params = PolicyParams(
            id=policy_id, kwargs=kwargs, entry_point=potmmcp_init_fn, info={}
        )
        policy_params.append(potmmcp_params)

    return policy_params


def tree_and_env_renderer_fn() -> Sequence[Renderer]:
    """Get environment and SearchTree renderers."""
    from potmmcp.run.render import SearchTreeRenderer
    return [EpisodeRenderer(), SearchTreeRenderer(1)]

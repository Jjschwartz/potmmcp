import copy
from itertools import product
from typing import List, Sequence, Dict, Union

import posggym.model as M

import baposgmcp.policy as P
import baposgmcp.tree as tree_lib
from baposgmcp.meta_policy import DictMetaPolicy
from baposgmcp.policy_prior import load_posggym_agents_policy_prior

from baposgmcp.run.exp import PolicyParams
from baposgmcp.run.render import Renderer, EpisodeRenderer


def baposgmcp_init_fn(model: M.POSGModel, agent_id: M.AgentID, kwargs):
    """Initialize BAPOSGMCP policy.

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
        model,
        agent_id,
        policy_prior_map=kwargs.pop("policy_prior_map")
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


def load_baposgmcp_params(variable_params: Dict[str, List],
                          baposgmcp_kwargs: Dict,
                          policy_prior_map: Union[
                              P.AgentPolicyDist,
                              Dict[P.PolicyState, float]
                          ],
                          meta_policy_dict: Dict[P.PolicyState, P.PolicyDist],
                          base_policy_id: str = "baposgmcp"
                          ) -> List[PolicyParams]:
    """Load list of policy params for BAPOSGMCP with different num sims."""
    base_kwargs = copy.deepcopy(baposgmcp_kwargs)

    base_kwargs.update({
        "policy_prior_map": policy_prior_map,
        "meta_policy_dict": meta_policy_dict
    })

    policy_params = []
    param_names = list(variable_params)
    param_value_lists = [variable_params[k] for k in param_names]
    for values in product(*param_value_lists):
        # need to do copy as kwargs is modified in baposgmcp init fn
        kwargs = copy.deepcopy(base_kwargs)
        policy_id = base_policy_id
        for k, v in zip(param_names, values):
            kwargs[k] = v
            # remove _ so it's easier to parse when doing analysis
            policy_id += f"_{k.replace('_', '')}{v}"
        kwargs["policy_id"] = policy_id

        baposgmcp_params = PolicyParams(
            id=policy_id,
            kwargs=kwargs,
            entry_point=baposgmcp_init_fn,
            info={}
        )
        policy_params.append(baposgmcp_params)

    return policy_params


def tree_and_env_renderer_fn() -> Sequence[Renderer]:
    """Get environment and SearchTree renderers."""
    from baposgmcp.run.render import SearchTreeRenderer
    return [
        EpisodeRenderer(),
        SearchTreeRenderer(1)
    ]

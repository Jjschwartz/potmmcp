"""Baseline policies for BAPOSGMCP experiments."""
from typing import List, Dict, Optional

import baposgmcp.policy as P
from baposgmcp.run.exp import PolicyParams

from baposgmcp.baselines.mcp import load_fixed_pi_baposgmcp_params
from baposgmcp.baselines.mcp import load_random_baposgmcp_params
from baposgmcp.baselines.meta import MetaBaselinePolicy
from baposgmcp.baselines.po_meta import POMeta
from baposgmcp.baselines.po_meta import load_pometa_params
from baposgmcp.baselines.po_meta_rollout import POMetaRollout
from baposgmcp.baselines.po_meta_rollout import load_pometarollout_params


def load_all_baselines(variable_params: Dict[str, List],
                       baposgmcp_kwargs: Dict,
                       other_policy_dist: P.AgentPolicyDist,
                       meta_policy_dict: Dict[
                           P.PolicyState, P.PolicyDist
                       ],
                       policy_id_suffix: Optional[str] = None
                       ) -> List[PolicyParams]:
    """Load all baseline policy params."""
    if "num_sims" in variable_params:
        num_sims = variable_params["num_sims"]
    elif "num_sims" in baposgmcp_kwargs:
        num_sims = [baposgmcp_kwargs["num_sims"]]
    else:
        raise ValueError("'num_sims' parameter must be defined")

    if not policy_id_suffix:
        policy_id_suffix = ""
    elif not policy_id_suffix.startswith("_"):
        policy_id_suffix = f"_{policy_id_suffix}"

    baseline_params = []

    # Meta Baseline Policy
    policy_id = f"metabaseline{policy_id_suffix}"
    policy_params = PolicyParams(
        id=policy_id,
        entry_point=MetaBaselinePolicy.posggym_agents_entry_point,
        kwargs={
            "policy_id": policy_id,
            "other_policy_dist": other_policy_dist,
            "meta_policy_dict": meta_policy_dict
        }
    )
    baseline_params.append(policy_params)

    # PO-Meta for different number of simulations
    pometa_params = load_pometa_params(
        num_sims,
        other_policy_dist=other_policy_dist,
        meta_policy_dict=meta_policy_dict,
        kwargs=baposgmcp_kwargs,
        base_policy_id=f"POMeta{policy_id_suffix}"
    )
    baseline_params.extend(pometa_params)

    # # POMetaRollout policies
    pometarollout_params = load_pometarollout_params(
        variable_params,
        baposgmcp_kwargs=baposgmcp_kwargs,
        other_policy_dist=other_policy_dist,
        meta_policy_dict=meta_policy_dict,
        base_policy_id=f"POMetaRollout{policy_id_suffix}"
    )
    baseline_params.extend(pometarollout_params)

    return baseline_params

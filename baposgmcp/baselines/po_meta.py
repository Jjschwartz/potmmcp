import copy
import time
import random
from typing import List, Dict, Union

import posggym.model as M

import baposgmcp.policy as P
from baposgmcp.run.exp import PolicyParams
from baposgmcp.meta_policy import MetaPolicy, DictMetaPolicy
from baposgmcp.policy_prior import (
    PolicyPrior, load_posggym_agents_policy_prior
)

from baposgmcp.tree.policy import BAPOSGMCP
from baposgmcp.tree.reinvigorate import BeliefReinvigorator


def load_pometa_params(num_sims: List[int],
                       policy_prior_map: Union[
                           P.AgentPolicyDist,
                           Dict[P.PolicyState, float]
                       ],
                       meta_policy_dict: Dict[P.PolicyState, P.PolicyDist],
                       kwargs: Dict,
                       base_policy_id: str = "POMeta"
                       ) -> List[PolicyParams]:
    """Load list of policy params for POMeta.

    Returns policy params for each number of sims.

    """
    policy_params = []
    for n in num_sims:
        kwargs_n = copy.deepcopy(kwargs)
        # remove _ from 'num_sims' so it's easier to parse when doing analysis
        policy_id = f"{base_policy_id}_numsims{n}"
        kwargs_n.update({
            "policy_id": policy_id,
            "num_sims": n,
            "policy_prior_map": policy_prior_map,
            "meta_policy_dict": meta_policy_dict,
        })
        params = PolicyParams(
            id=policy_id,
            entry_point=POMeta.posggym_agents_entry_point,
            kwargs=kwargs_n
        )
        policy_params.append(params)

    return policy_params


class POMeta(BAPOSGMCP):
    """Partially Observable Meta algorithm.

    Uses Monte-Carlo Belief state updates same as BAPOSGMCP and uses this
    to compute the policy for the next step by taking an expectation over the
    Meta-Policy w.r.t the belief over policy-states.

    """

    def __init__(self,
                 model: M.POSGModel,
                 agent_id: M.AgentID,
                 num_sims: int,
                 other_policy_prior: PolicyPrior,
                 meta_policy: MetaPolicy,
                 **kwargs):
        super().__init__(
            model,
            agent_id,
            discount=kwargs.pop("discount", 0.99),
            # belief size is based on num_sims
            num_sims=num_sims,
            other_policy_prior=other_policy_prior,
            meta_policy=meta_policy,
            c=kwargs.pop("c_init", 1.25),
            truncated=kwargs.pop("truncated", True),
            reinvigorator=kwargs.pop("reinvigorator", None),
            step_limit=kwargs.pop("step_limit", None),
            epsilon=kwargs.pop("epsilon", 0.01),
            policy_id=kwargs.pop("policy_id", "POMeta"),
            **kwargs
        )

    def update(self, action: M.Action, obs: M.Observation) -> None:
        super().update(action, obs)

        # update root node policy based on policy state belief
        policy_state_dist = self.root.belief.get_policy_state_dist()
        policy_dist = self._meta_policy.get_exp_policy_dist(policy_state_dist)
        policy = self._meta_policy.get_exp_action_dist(
            policy_dist,
            self.root.rollout_hidden_states
        )

        self.root.policy = policy
        if len(self.root.children) == 0:
            self._expand(self.root, self.history)

    def get_action(self) -> M.Action:
        self._log_info1(f"Searching for num_sims={self.num_sims}")
        start_time = time.time()

        max_search_depth = 0
        if self.root.is_absorbing:
            self._log_debug("Agent in absorbing state. Not running search.")
            action = self.root.children[0].action
        else:
            # select based on expected meta policy
            # which is added to root obs node during update
            action = random.choices(
                list(self.root.policy.keys()),
                weights=list(self.root.policy.values()),
                k=1
            )[0]

        search_time = time.time() - start_time
        search_time_per_sim = search_time / self.num_sims
        self._statistics["search_time"] = search_time
        self._statistics["search_depth"] = max_search_depth
        self._log_info1(
            f"{search_time=:.2f} {search_time_per_sim=:.5f} "
            f"{max_search_depth=}"
        )
        return action

    @staticmethod
    def posggym_agents_entry_point(model: M.POSGModel,
                                   agent_id: M.AgentID,
                                   kwargs):
        """Initialize POMeta Policy.

        Required kwargs
        ---------------
        num_sims: int,
        policy_prior_map: Union[P.AgentPolicyDist,Dict[P.PolicyState, float]]
        meta_policy_dict : Dict[P.PolicyState, P.PolicyDist]

        Optional kwargs
        ---------------
        policy_id : str
        reinvigorator: BeliefReinvigorator

        """
        kwargs = copy.deepcopy(kwargs)

        policy_prior = load_posggym_agents_policy_prior(
            model,
            agent_id,
            policy_prior_map=kwargs.pop("policy_prior_map")
        )

        meta_policy = DictMetaPolicy.load_possgym_agents_meta_policy(
            model,
            agent_id,
            meta_policy_dict=kwargs.pop("meta_policy_dict")
        )

        return POMeta(
            model,
            agent_id,
            other_policy_prior=policy_prior,
            meta_policy=meta_policy,
            **kwargs,
        )

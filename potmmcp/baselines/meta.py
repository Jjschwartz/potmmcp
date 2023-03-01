import copy
from typing import Optional

import posggym.model as M
from posggym.utils.history import AgentHistory
from posggym_agents.policy import ActionDist, BasePolicy

from potmmcp.meta_policy import DictMetaPolicy, MetaPolicy
from potmmcp.policy_prior import PolicyPrior, load_posggym_agents_policy_prior


class MetaBaselinePolicy(BasePolicy):
    """Baseline policy that selects policy to use given policy-prior.

    For each episode a policy state is sampled from the prior and then the
    fixed policy to use for the episode is sampled from the Meta-Policy.
    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: M.AgentID,
        policy_id: str,
        policy_prior: PolicyPrior,
        meta_policy: MetaPolicy,
    ):
        super().__init__(model, agent_id, policy_id)
        self.policy_prior = policy_prior
        self.meta_policy = meta_policy

        self._current_policy_state = policy_prior.sample_policy_state()
        self._current_policy = meta_policy.sample(self._current_policy_state)

    def get_action(self) -> M.Action:
        return self._current_policy.get_action()

    def update(self, action: M.Action, obs: M.Observation) -> None:
        super().update(action, obs)
        self._current_policy.update(action, obs)

    def get_pi(self, history: Optional[AgentHistory] = None) -> ActionDist:
        return self._current_policy.get_pi(history)

    def reset(self) -> None:
        super().reset()
        policy_state = self.policy_prior.sample_policy_state()
        self._current_policy_state = policy_state
        self._current_policy = self.meta_policy.sample(policy_state)
        self._current_policy.reset()

    def reset_history(self, history: AgentHistory) -> None:
        super().reset_history(history)
        self._current_policy.reset_history(history)

    @staticmethod
    def posggym_agents_entry_point(model: M.POSGModel, agent_id: M.AgentID, kwargs):
        """Initialize MetaBaselinePolicy.

        Required kwargs
        ---------------
        policy_prior_map : Union[P.AgentPolicyDist, Dict[P.PolicyState, float]]
        meta_policy_dict : Dict[P.PolicyState, P.PolicyDist]

        Optional kwargs
        ---------------
        policy_id : str

        """
        kwargs = copy.deepcopy(kwargs)

        policy_prior = load_posggym_agents_policy_prior(
            model, agent_id, policy_prior_map=kwargs["policy_prior_map"]
        )

        meta_policy = DictMetaPolicy.load_possgym_agents_meta_policy(
            model, agent_id, meta_policy_dict=kwargs["meta_policy_dict"]
        )

        return MetaBaselinePolicy(
            model,
            agent_id,
            kwargs.get("policy_id", "metabaseline"),
            policy_prior,
            meta_policy,
        )

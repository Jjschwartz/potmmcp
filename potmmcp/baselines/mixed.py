import random
from typing import Dict, Optional, Tuple

import posggym.model as M
import posggym_agents
from posggym.utils.history import AgentHistory
from posggym_agents.policy import ActionDist, BasePolicy


class MixedPolicy(BasePolicy):
    """Policy that selects policy to use based on policy distribution.

    Each episode the policy to use for the episode is sampled from the policy
    distribution.
    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: M.AgentID,
        policy_id: str,
        policy_dist: Dict[str, Tuple[BasePolicy, float]],
    ):
        super().__init__(model, agent_id, policy_id)
        self.policy_dist = policy_dist
        self._policies = [v[0] for v in policy_dist.values()]
        self._policy_probs = [v[1] for v in policy_dist.values()]
        self._current_policy = self._sample_policy()

    def _sample_policy(self) -> BasePolicy:
        return random.choices(self._policies, weights=self._policy_probs, k=1)[0]

    def get_action(self) -> M.Action:
        return self._current_policy.get_action()

    def update(self, action: M.Action, obs: M.Observation) -> None:
        super().update(action, obs)
        self._current_policy.update(action, obs)

    def get_pi(self, history: Optional[AgentHistory] = None) -> ActionDist:
        return self._current_policy.get_pi(history)

    def reset(self) -> None:
        super().reset()
        self._current_policy = self._sample_policy()
        self._current_policy.reset()

    def reset_history(self, history: AgentHistory) -> None:
        super().reset_history(history)
        self._current_policy.reset_history(history)

    @staticmethod
    def posggym_agents_entry_point(model: M.POSGModel, agent_id: M.AgentID, kwargs):
        """Initialize MetaBaselinePolicy.

        Required kwargs
        ---------------
        policy_dist : Dict[str, float]

        Optional kwargs
        ---------------
        policy_id : str

        """
        policy_dist = {}
        for pi_id, prob in kwargs["policy_dist"].items():
            policy = posggym_agents.make(pi_id, model, agent_id)
            policy_dist[pi_id] = (policy, prob)

        return MixedPolicy(
            model,
            agent_id,
            kwargs.get("policy_id", "mixed"),
            policy_dist,
        )

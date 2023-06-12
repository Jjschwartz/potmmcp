import random
from typing import Optional, List

from gym import spaces

import posggym.model as M
from posggym.utils.history import AgentHistory

import posggym_agents.policy as Pi


class FixedDistributionPolicy(Pi.BaseHiddenStatePolicy):
    """A policy that samples from a fixed distribution."""

    def __init__(self,
                 model: M.POSGModel,
                 agent_id: M.AgentID,
                 policy_id: Pi.PolicyID,
                 dist: Pi.ActionDist):
        super().__init__(model, agent_id, policy_id)

        action_space = self.model.action_spaces[self.agent_id]
        assert isinstance(action_space, spaces.Discrete)

        self._action_space = list(range(action_space.n))
        self._dist = dist
        self._cum_weights: List[float] = []
        for i, a in enumerate(self._action_space):
            prob_sum = 0.0 if i == 0 else self._cum_weights[-1]
            self._cum_weights.append(self._dist[a] + prob_sum)

    def get_action(self) -> M.Action:
        return random.choices(
            self._action_space, cum_weights=self._cum_weights, k=1
        )[0]

    def get_action_by_hidden_state(self,
                                   hidden_state: Pi.PolicyHiddenState
                                   ) -> M.Action:
        return self.get_action()

    def get_pi(self,
               history: Optional[AgentHistory] = None
               ) -> Pi.ActionDist:
        return dict(self._dist)

    def get_pi_from_hidden_state(self,
                                 hidden_state: Pi.PolicyHiddenState
                                 ) -> Pi.ActionDist:
        return self.get_pi(None)

    def get_value(self, history: Optional[AgentHistory]) -> float:
        raise NotImplementedError

    def get_value_by_hidden_state(self,
                                  hidden_state: Pi.PolicyHiddenState) -> float:
        raise NotImplementedError


class RandomPolicy(FixedDistributionPolicy):
    """Uniform Random policy."""

    def __init__(self,
                 model: M.POSGModel,
                 agent_id: M.AgentID,
                 policy_id: str):
        action_space = model.action_spaces[agent_id]
        super().__init__(
            model,
            agent_id,
            policy_id,
            dist={a: 1.0 / action_space.n for a in range(action_space.n)}
        )

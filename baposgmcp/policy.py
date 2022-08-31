"""Base policy class and utility functions."""
import abc
import random
import logging
from typing import Optional, Dict, Mapping, Any, Union, Tuple, List

import gym
import posggym.model as M

from baposgmcp import parts
from baposgmcp.history import AgentHistory

# Convenient type definitions
PolicyID = Union[int, str]
ActionDist = Dict[M.Action, float]
PolicyMap = Dict[PolicyID, "BasePolicy"]
PolicyState = Tuple[PolicyID, ...]
PolicyHiddenState = Dict[str, Any]
PolicyHiddenStates = Tuple[PolicyHiddenState, ...]
PolicyHiddenStateMap = Dict[PolicyID, PolicyHiddenState]
PolicyDist = Dict[PolicyID, float]
PolicyVectorDist = Tuple[Tuple[PolicyID, ...], Tuple[float, ...]]
AgentPolicyMap = Dict[parts.AgentID, PolicyMap]
AgentPolicyDist = Dict[parts.AgentID, PolicyDist]


def sample_action_dist(dist: ActionDist) -> M.Action:
    """Sample an action from an action distribution."""
    return random.choices(
        list(dist.keys()), weights=list(dist.values()), k=1
    )[0]


def vectorize_policy_dist(dist: PolicyDist,
                          make_cumulative: bool) -> PolicyVectorDist:
    """Convert a policy distribution dictionary into a vectorized format.

    The vectorized format is two tuples, one for policyIDs and one for the
    corresponding probabilities.

    If make_cumulative=True, distribution vector will have cumulative
    probabilities, otherwise they are just normal. Cumulative probabilities are
    faster when working with random.choices function.
    """
    policy_ids = tuple(dist)
    policy_probs = tuple(dist.values())

    prob_sum = sum(policy_probs)
    is_cumulative = prob_sum >= 1.0 and policy_probs[-1] == 1.0

    if not is_cumulative and prob_sum != 1.0:
        policy_probs = parts.normalize_dist(policy_probs)

    if make_cumulative and not is_cumulative:
        cumulative_probs = []
        last_prob = 0.0
        for p in policy_probs:
            last_prob = p + last_prob
            cumulative_probs.append(last_prob)
        policy_probs = tuple(cumulative_probs)
    elif not make_cumulative and is_cumulative:
        noncumulative_policy_probs = []
        last_prob = 0
        for p in policy_probs:
            noncumulative_policy_probs.append(p-last_prob)
            last_prob = p
        policy_probs = tuple(noncumulative_policy_probs)

    return policy_ids, policy_probs


class BasePolicy(abc.ABC):
    """Abstract policy interface."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 policy_id: Optional[str],
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        self.model = model
        self.ego_agent = ego_agent
        self.gamma = gamma
        self.policy_id = policy_id
        self._logger = logging.getLogger() if logger is None else logger
        self.kwargs = kwargs

        self.history = AgentHistory.get_init_history()
        self._last_action = None
        self._statistics: Dict[str, Any] = {}

    def step(self, obs: M.Observation) -> M.Action:
        """Execute a single policy step.

        This involves:
        1. a updating policy with last action and given observation
        2. next action using updated policy
        """
        self.update(self._last_action, obs)
        self._last_action = self.get_action()
        return self._last_action

    @abc.abstractmethod
    def get_action(self) -> M.Action:
        """Get action for given obs."""

    def get_action_by_history(self, history: AgentHistory) -> M.Action:
        """Get action given history, leaving state of policy unchanged."""
        current_history = self.history
        self.reset_history(history)
        action = self.get_action()
        self.reset_history(current_history)
        return action

    @abc.abstractmethod
    def get_action_by_hidden_state(self,
                                   hidden_state: PolicyHiddenState
                                   ) -> M.Action:
        """Get action given hidden state of policy."""

    @abc.abstractmethod
    def get_pi(self,
               history: Optional[AgentHistory] = None
               ) -> ActionDist:
        """Get agent's distribution over actions for a given history.

        If history is None or not given then uses current history.
        """

    @abc.abstractmethod
    def get_pi_from_hidden_state(self,
                                 hidden_state: PolicyHiddenState
                                 ) -> ActionDist:
        """Get agent's distribution over actions for given hidden state."""

    @abc.abstractmethod
    def get_value(self, history: Optional[AgentHistory]) -> float:
        """Get a value estimate of a history."""

    @abc.abstractmethod
    def get_value_by_hidden_state(self,
                                  hidden_state: PolicyHiddenState) -> float:
        """Get a value estimate from policy's hidden state."""

    def update(self, action: M.Action, obs: M.Observation) -> None:
        """Update policy history."""
        self.history = self.history.extend(action, obs)

    def reset(self) -> None:
        """Reset the policy."""
        self.history = AgentHistory.get_init_history()
        self._last_action = None

    def reset_history(self, history: AgentHistory) -> None:
        """Reset policy history to given history."""
        self.history = history

    def get_next_hidden_state(self,
                              hidden_state: PolicyHiddenState,
                              action: M.Action,
                              obs: M.Observation
                              ) -> PolicyHiddenState:
        """Get next hidden state of policy."""
        if hidden_state["history"] is None:
            next_history = AgentHistory(((action, obs), ))
        else:
            next_history = hidden_state["history"].extend(action, obs)
        return {
            "history": next_history,
            "last_action": action
        }

    def get_initial_hidden_state(self) -> PolicyHiddenState:
        """Get the initial hidden state of the policy."""
        return {
            "history": None,
            "last_action": None
        }

    def get_hidden_state(self) -> PolicyHiddenState:
        """Get the hidden state of the policy given it's current history."""
        return {
            "history": self.history,
            "last_action": self._last_action
        }

    def set_hidden_state(self, hidden_state: PolicyHiddenState):
        """Set the hidden state of the policy."""
        self.history = hidden_state["history"]
        self._last_action = hidden_state["last_action"]

    #######################################################
    # Logging
    #######################################################

    @property
    def statistics(self) -> Mapping[str, Any]:
        """Return current agent statistics as a dictionary."""
        return self._statistics

    def _log_info1(self, msg: str):
        """Log an info message."""
        self._logger.log(logging.INFO - 1, self._format_msg(msg))

    def _log_info2(self, msg: str):
        """Log an info message."""
        self._logger.log(logging.INFO - 2, self._format_msg(msg))

    def _log_debug(self, msg: str):
        """Log a debug message."""
        self._logger.debug(self._format_msg(msg))

    def _log_debug1(self, msg: str):
        """Log a debug2 message."""
        self._logger.log(logging.DEBUG - 1, self._format_msg(msg))

    def _log_debug2(self, msg: str):
        """Log a debug2 message."""
        self._logger.log(logging.DEBUG - 2, self._format_msg(msg))

    def _format_msg(self, msg: str):
        return f"i={self.ego_agent} {msg}"

    def __str__(self):
        return self.__class__.__name__


class FixedDistributionPolicy(BasePolicy):
    """A policy that samples from a fixed distribution."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 dist: ActionDist,
                 policy_id: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        super().__init__(
            model,
            ego_agent,
            gamma,
            policy_id=policy_id,
            logger=logger
        )
        action_space = self.model.action_spaces[self.ego_agent]
        assert isinstance(action_space, gym.spaces.Discrete)
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

    def get_pi(self,
               history: Optional[AgentHistory] = None
               ) -> ActionDist:
        return dict(self._dist)

    def get_pi_from_hidden_state(self,
                                 hidden_state: PolicyHiddenState
                                 ) -> ActionDist:
        return self.get_pi(None)

    def get_value(self, history: Optional[AgentHistory]) -> float:
        return 0.0

    def get_value_by_hidden_state(self,
                                  hidden_state: PolicyHiddenState) -> float:
        return 0.0

    def get_action_by_hidden_state(self,
                                   hidden_state: PolicyHiddenState
                                   ) -> M.Action:
        return self.get_action()


class RandomPolicy(FixedDistributionPolicy):
    """Random policy."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 policy_id: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        action_space = model.action_spaces[ego_agent]
        super().__init__(
            model,
            ego_agent,
            gamma,
            dist={a: 1.0 / action_space.n for a in range(action_space.n)},
            policy_id="pi_-1" if policy_id is None else policy_id,
            logger=logger
        )

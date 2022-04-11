"""Base policy class and utility functions."""
import abc
import random
import logging
from typing import Optional, Dict, Mapping, Any, Tuple

import gym
import posggym.model as M

from baposgmcp import parts
import baposgmcp.hps as H


class BasePolicy(abc.ABC):
    """Abstract policy interface."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        self.model = model
        self.ego_agent = ego_agent
        self.gamma = gamma
        self._logger = logging.getLogger() if logger is None else logger
        self.kwargs = kwargs

        self.history = H.AgentHistory.get_init_history()
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

    def get_action_by_history(self, history: H.AgentHistory) -> M.Action:
        """Get action given history, leaving state of policy unchanged."""
        current_history = self.history
        self.reset_history(history)
        action = self.get_action()
        self.reset_history(current_history)
        return action

    @abc.abstractmethod
    def get_pi(self,
               history: Optional[H.AgentHistory] = None
               ) -> parts.ActionDist:
        """Get agent's distribution over actions for a given history.

        If history is None or not given then uses current history.
        """

    def update(self, action: M.Action, obs: M.Observation) -> None:
        """Update policy history."""
        self.history = self.history.extend(action, obs)

    def reset(self) -> None:
        """Reset the policy."""
        self.history = H.AgentHistory.get_init_history()
        self._last_action = None

    def reset_history(self, history: H.AgentHistory) -> None:
        """Reset policy history to given history."""
        self.history = history

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


class BaseRolloutPolicy(BasePolicy, abc.ABC):
    """Abstract rollout policy interface.

    A rollout policy adds some additional methods to the base policy class such
    as getting the initial values, etc..
    """

    @abc.abstractmethod
    def get_initial_action_values(self,
                                  history: H.AgentHistory
                                  ) -> Dict[M.Action, Tuple[float, int]]:
        """Get initial values and visit count for each action for a history."""

    @abc.abstractmethod
    def get_value(self, history: Optional[H.AgentHistory]) -> float:
        """Get a value estimate of a history."""


class FixedDistributionPolicy(BasePolicy):
    """A policy that samples from a fixed distribution."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 dist: parts.ActionDist,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        super().__init__(model, ego_agent, gamma, logger)
        action_space = self.model.action_spaces[self.ego_agent]
        assert isinstance(action_space, gym.spaces.Discrete)
        self._action_space = list(range(action_space.n))
        self._dist = dist
        self._cum_weights = []
        for i, a in enumerate(self._action_space):
            prob_sum = 0.0 if i == 0 else self._cum_weights[-1]
            self._cum_weights.append(self._dist[a] + prob_sum)

    def get_action(self) -> M.Action:
        return random.choices(
            self._action_space, cum_weights=self._cum_weights, k=1
        )[0]

    def get_pi(self,
               history: Optional[H.AgentHistory] = None
               ) -> parts.ActionDist:
        return dict(self._dist)


class RandomPolicy(FixedDistributionPolicy):
    """Random policy."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        action_space = model.action_spaces[ego_agent]
        super().__init__(
            model,
            ego_agent,
            gamma,
            dist={a: 1.0 / action_space.n for a in range(action_space.n)},
            logger=logger
        )


class FixedDistributionRolloutPolicy(BaseRolloutPolicy):
    """A policy that samples from a fixed distribution."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 dist: parts.ActionDist,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        super().__init__(model, ego_agent, gamma, logger)
        action_space = self.model.action_spaces[self.ego_agent]
        assert isinstance(action_space, gym.spaces.Discrete)
        self._action_space = list(range(action_space.n))
        self._dist = dist
        self._cum_weights = []
        for i, a in enumerate(self._action_space):
            prob_sum = 0.0 if i == 0 else self._cum_weights[-1]
            self._cum_weights.append(self._dist[a] + prob_sum)

    def get_action(self) -> M.Action:
        return random.choices(
            self._action_space, cum_weights=self._cum_weights, k=1
        )[0]

    def get_initial_action_values(self,
                                  history: H.AgentHistory
                                  ) -> Dict[M.Action, Tuple[float, int]]:
        return {a: (0.0, 0) for a in self._action_space}

    def get_value(self, history: Optional[H.AgentHistory]) -> float:
        return 0.0

    def get_pi(self,
               history: Optional[H.AgentHistory] = None
               ) -> parts.ActionDist:
        return dict(self._dist)


class RandomRolloutPolicy(FixedDistributionRolloutPolicy):
    """Uniform random rollout policy."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        action_space = model.action_spaces[ego_agent]
        super().__init__(
            model,
            ego_agent,
            gamma,
            dist={a: 1.0 / action_space.n for a in range(action_space.n)},
            logger=logger
        )

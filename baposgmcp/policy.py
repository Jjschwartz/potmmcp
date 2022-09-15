"""Base policy class and utility functions."""
import abc
import logging
from typing import Optional, Dict, Mapping, Any, Tuple

import posggym.model as M
from posggym.utils.history import AgentHistory
from posggym_agents.policy import (
    BasePolicy, BaseHiddenStatePolicy, PolicyID, PolicyHiddenState
)

# Convenient type definitions
ActionDist = Dict[M.Action, float]
PolicyMap = Dict[PolicyID, "BasePolicy"]
PolicyState = Tuple[PolicyID, ...]
PolicyHiddenStates = Tuple[PolicyHiddenState, ...]
PolicyHiddenStateMap = Dict[PolicyID, PolicyHiddenState]
PolicyDist = Dict[PolicyID, float]
PolicyVectorDist = Tuple[Tuple[PolicyID, ...], Tuple[float, ...]]
AgentPolicyMap = Dict[M.AgentID, PolicyMap]
AgentPolicyDist = Dict[M.AgentID, PolicyDist]


class BAPOSGMCPBasePolicy(BaseHiddenStatePolicy, abc.ABC):
    """Base policy for BAPOSGMCP.

    Adds some attributes to the posggym_agents.BasePolicy class.
    """

    def __init__(self,
                 model: M.POSGModel,
                 agent_id: int,
                 policy_id: str,
                 discount: float,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        super().__init__(model, agent_id, policy_id)
        self.discount = discount
        self._logger = logging.getLogger() if logger is None else logger
        self.kwargs = kwargs
        self._statistics: Dict[str, Any] = {}

    def get_action_by_history(self, history: AgentHistory) -> M.Action:
        """Get action given history, leaving state of policy unchanged."""
        current_history = self.history
        self.reset_history(history)
        action = self.get_action()
        self.reset_history(current_history)
        return action

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
        return f"i={self.agent_id} {msg}"

    def __str__(self):
        return self.__class__.__name__

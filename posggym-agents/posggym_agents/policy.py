import abc
from typing import Optional, Dict, Any

import posggym.model as M
from posggym.utils.history import AgentHistory


# Convenient type definitions
PolicyID = str
ActionDist = Dict[M.Action, float]
PolicyHiddenState = Dict[str, Any]


class BasePolicy(abc.ABC):
    """Abstract policy interface.

    Subclasses must implement:

    get_action
    get_pi

    Subclasses may additionally wish to implement:

    get_value

    Subclasses will likely need to override depending on their implementation:

    reset
    update
    reset_history

    """

    # PolicySpec used to generate policy instance
    # This is set when policy is made using make function
    spec = None

    def __init__(self,
                 model: M.POSGModel,
                 agent_id: M.AgentID,
                 policy_id: str):
        self.model = model
        self.agent_id = agent_id
        self.policy_id = policy_id
        self.history = AgentHistory.get_init_history()
        self._last_action = None

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
        """Get action given agent's current history."""

    def get_action_by_history(self, history: AgentHistory) -> M.Action:
        """Get action given history, leaving state of policy unchanged."""
        current_history = self.history
        self.reset_history(history)
        action = self.get_action()
        self.reset_history(current_history)
        return action

    @abc.abstractmethod
    def get_pi(self,
               history: Optional[AgentHistory] = None
               ) -> ActionDist:
        """Get agent's distribution over actions for a given history.

        If history is None then uses current history.
        """

    def get_value(self, history: Optional[AgentHistory]) -> float:
        """Get a value estimate of a history."""
        raise NotImplementedError(
            f"get_value() function not supported by {self.__class__.__name__}"
        )

    def update(self, action: M.Action, obs: M.Observation) -> None:
        """Update policy."""
        self.history = self.history.extend(action, obs)

    def reset(self) -> None:
        """Reset the policy to it's initial state."""
        self.history = AgentHistory.get_init_history()
        self._last_action = None

    def reset_history(self, history: AgentHistory) -> None:
        """Reset policy state based on a history."""
        self.history = history


class BaseHiddenStatePolicy(BasePolicy, abc.ABC):
    """Abstract Hidden State policy interface.

    Adds additional functions for accessing and setting the internal hidden
    state of a policy.

    Subclasses need to implement:

    get_action_by_hidden_state
    get_pi_from_hidden_state

    Subclasse may wish to implement:

    get_value_by_hidden_state

    Additional, subclasses can and sometime should override:

    get_next_hidden_state
    get_initial_hidden_state
    get_hidden_state
    set_hidden_state

    If overriding these methods subclasses should call super() to ensure
    history and last_action attributes are part of the hidden state.

    """

    @abc.abstractmethod
    def get_action_by_hidden_state(self,
                                   hidden_state: PolicyHiddenState
                                   ) -> M.Action:
        """Get action given hidden state of agent."""

    @abc.abstractmethod
    def get_pi_from_hidden_state(self,
                                 hidden_state: PolicyHiddenState
                                 ) -> ActionDist:
        """Get agent's distribution over actions for given hidden state."""

    def get_value_by_hidden_state(self,
                                  hidden_state: PolicyHiddenState) -> float:
        """Get a value estimate from policy's hidden state."""
        raise NotImplementedError(
            "get_value_by_hidden_state() function not supported by "
            f"{self.__class__.__name__}"
        )

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

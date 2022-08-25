from typing import Tuple, Optional

import posggym.model as M

import baposgmcp.policy as P
import baposgmcp.history as H


PolicyState = Tuple[P.PolicyID, ...]


class HistoryPolicyState:
    """A History Policy state.

    Consists of:
    1. the environment state
    2. the joint history of each agent up to given state
    3. the policies of the other agents
    4. (Optional) the hidden/internal state of policies

    The hidden/internal state is not a defining part of the HistoryPolicyState,
    since the it can in general be obtained from the history contained in the
    HistoryPolicyState. Rather it is potentially stored for to save
    recomputing it, which can offer significan saving in some cases (e.g. when
    dealing with a policy that uses an RNN).
    """

    def __init__(self,
                 state: M.State,
                 history: H.JointHistory,
                 other_policies: PolicyState,
                 hidden_states: Optional[P.PolicyHiddenStates] = None):
        super().__init__()
        assert (
            hidden_states is None or len(hidden_states) == len(other_policies)
        )
        self.state = state
        self.history = history
        self.other_policies = other_policies
        self.hidden_states = hidden_states

    def __hash__(self):
        return hash((self.state, self.history, self.other_policies))

    def __eq__(self, other):
        if not isinstance(other, HistoryPolicyState):
            return False
        return (
            self.state == other.state
            and self.history == other.history
            and self.other_policies == other.other_policies
        )

    def __str__(self):
        return f"[s={self.state}, h={self.history}, pi={self.other_policies}]"

    def __repr__(self):
        return self.__str__()

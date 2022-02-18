"""A state in the Bayes Adaptive POSGMCP algorithm

History Policy state is made up of:
1. the environment state
2. the joint history of each agent up to given state
3. the policies of the other agents
"""
from typing import Tuple

import posggym.model as M

import posgmcp.history as H


PolicyID = int
PolicyState = Tuple[PolicyID, ...]


class HistoryPolicyState:
    """A History Policy state

    Consists of:
    1. the environment state
    2. the joint history of each agent up to given state
    3. the policies of the other agents
    """

    def __init__(self,
                 state: M.State,
                 history: H.JointHistory,
                 other_policies: PolicyState):
        super().__init__()
        self.state = state
        self.history = history
        self.other_policies = other_policies

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

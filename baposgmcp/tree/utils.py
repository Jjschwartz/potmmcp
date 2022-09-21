from typing import Optional, Dict

import posggym.model as M
from posggym.utils.history import AgentHistory

import baposgmcp.policy as P
from baposgmcp.tree.policy import BAPOSGMCP


def get_state_belief(tree: BAPOSGMCP) -> Optional[Dict[M.State, float]]:
    """Get tree's root node distribution over states.

    May return distribution over only states with p(s) > 0 as opposed to all
    environment states.

    Returns None if belief is empty for given history. This can occur if the
    agent (represented by the tree policy) has reached an absorbing/terminal
    state, yet the environment episode has not terminated.
    """
    if tree.root.belief.size() == 0:
        return None

    state_belief = {}
    for hp_state, prob in tree.root.belief.get_dist().items():
        state = hp_state.state   # type: ignore
        if state not in state_belief:
            state_belief[state] = 0.0
        state_belief[state] += prob
    return state_belief


def get_other_pis_belief(tree: BAPOSGMCP) -> Optional[P.AgentPolicyDist]:
    """Get tree's root node belief over other agent policies.

    This returns a distribution over policies for each opponent/other agent
    in the environment.

    Returns None if belief is empty for given history. This can occur if the
    agent (represented by the tree policy) has reached an absorbing/terminal
    state, yet the environment episode has not terminated.
    """
    if tree.root.belief.size() == 0:
        return None

    pi_belief: P.AgentPolicyDist = {}
    # pylint: disable=[protected-access]
    other_policies = tree._other_policy_prior.policies
    for i in range(tree.num_agents):
        if i == tree.agent_id:
            continue
        pi_belief[i] = {pi_id: 0.0 for pi_id in other_policies[i]}

    for pi_state, prob in tree.root.belief.get_policy_state_dist().items():
        for i in range(tree.num_agents):
            if i == tree.agent_id:
                continue
            pi_id = pi_state[i]
            pi_belief[i][pi_id] += prob

    return pi_belief


def get_other_history_belief(tree: BAPOSGMCP
                             ) -> Optional[
                                 Dict[M.AgentID, Dict[AgentHistory, float]]
                             ]:
    """Get tree's root node belief over history of other agents.

    Returns None if belief is empty for given history. This can occur if the
    agent (represented by the tree policy) has reached an absorbing/terminal
    state, yet the environment episode has not terminated.
    """
    if tree.root.belief.size() == 0:
        return None

    history_belief: Dict[M.AgentID, Dict[AgentHistory, float]] = {}
    for i in range(tree.num_agents):
        if i == tree.agent_id:
            continue
        history_belief[i] = {}

    for hp_state, prob in tree.root.belief.get_dist().items():
        h = hp_state.history    # type: ignore
        for i in range(tree.num_agents):
            if i == tree.agent_id:
                continue
            h_i = h.get_agent_history(i)
            if h_i not in history_belief[i]:
                history_belief[i][h_i] = 0.0
            history_belief[i][h_i] += prob
    return history_belief


def get_other_agent_action_dist(tree: BAPOSGMCP
                                ) -> Optional[
                                    Dict[M.AgentID, P.ActionDist]
                                ]:
    """Get tree's root node belief over action dists of other agents.

    Returns None if belief is empty for given history. This can occur if the
    agent (represented by the tree policy) has reached an absorbing/terminal
    state, yet the environment episode has not terminated.
    """
    if tree.root.belief.size() == 0:
        return None

    action_belief: Dict[M.AgentID, Dict[M.Action, float]] = {}
    for i in range(tree.num_agents):
        if i == tree.agent_id:
            continue
        action_space_i = list(range(tree.model.action_spaces[i].n))
        action_belief[i] = {a: 0.0 for a in action_space_i}

    for hp_state, prob in tree.root.belief.get_dist().items():
        other_pis = tree.get_other_agent_pis(hp_state)
        for i in range(tree.num_agents):
            if i == tree.agent_id:
                continue
            pi_i = other_pis[i]
            for a_i in action_belief[i]:
                action_belief[i][a_i] += pi_i[a_i]

    # normalize
    for i in action_belief:
        prob_sum = sum(p_a_i for p_a_i in action_belief[i].values())
        for a_i in action_belief[i]:
            action_belief[i][a_i] /= prob_sum

    return action_belief

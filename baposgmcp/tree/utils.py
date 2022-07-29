from argparse import ArgumentParser
from typing import Optional, Dict

import posggym.model as M

import baposgmcp.hps as H
from baposgmcp import parts
import baposgmcp.tree.policy as tree_lib


def get_baposgmcp_args_parser(parser: Optional[ArgumentParser] = None
                              ) -> ArgumentParser:
    """Get argument parser for POSGMCP."""
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument(
        "--uct_c", type=float, default=None,
        help="UCT C Hyperparam, if None uses reward range (r_max-r_min)."
    )
    parser.add_argument(
        "--num_sims", nargs="*", type=int, default=[1024],
        help="Number of simulations to run"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Discount Horizon Threshold"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95,
        help="Discount"
    )
    return parser


def get_state_belief(tree: tree_lib.BAPOSGMCP) -> Optional[parts.StateDist]:
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


def get_other_pis_belief(tree: tree_lib.BAPOSGMCP
                         ) -> Optional[
                             Dict[M.AgentID, Dict[parts.PolicyID, float]]
                         ]:
    """Get tree's root node belief over other agent policies.

    This returns a distribution over policies for each opponent/other agent
    in the environment.

    Returns None if belief is empty for given history. This can occur if the
    agent (represented by the tree policy) has reached an absorbing/terminal
    state, yet the environment episode has not terminated.
    """
    if tree.root.belief.size() == 0:
        return None

    pi_belief: Dict[M.AgentID, Dict[parts.PolicyID, float]] = {}
    for i in range(tree.num_agents):
        if i == tree.ego_agent:
            continue
        # pylint: disable=[protected-access]
        pi_belief[i] = {pi_id: 0.0 for pi_id in tree._other_prior[i][0]}

    for hp_state, prob in tree.root.belief.get_dist().items():
        pi_state = hp_state.other_policies   # type: ignore
        for i in range(tree.num_agents):
            if i == tree.ego_agent:
                continue
            pi_id = pi_state[i]
            pi_belief[i][pi_id] += prob

    return pi_belief


def get_other_history_belief(tree: tree_lib.BAPOSGMCP
                             ) -> Optional[
                                 Dict[M.AgentID, Dict[H.AgentHistory, float]]
                             ]:
    """Get tree's root node belief over history of other agents.

    Returns None if belief is empty for given history. This can occur if the
    agent (represented by the tree policy) has reached an absorbing/terminal
    state, yet the environment episode has not terminated.
    """
    if tree.root.belief.size() == 0:
        return None

    history_belief: Dict[M.AgentID, Dict[H.AgentHistory, float]] = {}
    for i in range(tree.num_agents):
        if i == tree.ego_agent:
            continue
        history_belief[i] = {}

    for hp_state, prob in tree.root.belief.get_dist().items():
        h = hp_state.history    # type: ignore
        for i in range(tree.num_agents):
            if i == tree.ego_agent:
                continue
            h_i = h.get_agent_history(i)
            if h_i not in history_belief[i]:
                history_belief[i][h_i] = 0.0
            history_belief[i][h_i] += prob
    return history_belief


def get_other_agent_action_dist(tree: tree_lib.BAPOSGMCP
                                ) -> Optional[
                                    Dict[M.AgentID, parts.ActionDist]
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
        if i == tree.ego_agent:
            continue
        action_space_i = list(range(tree.model.action_spaces[i].n))
        action_belief[i] = {a: 0.0 for a in action_space_i}

    for hp_state, prob in tree.root.belief.get_dist().items():
        other_pis = tree.get_other_agent_pis(hp_state)
        for i in range(tree.num_agents):
            if i == tree.ego_agent:
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

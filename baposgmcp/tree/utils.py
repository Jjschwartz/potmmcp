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


def get_belief_by_history(tree: tree_lib.BAPOSGMCP,
                          history: Optional[H.AgentHistory] = None
                          ) -> parts.StateDist:
    """Get agent's distribution over states for a given history.

    May return distribution over only states with p(s) > 0
    """
    if history is None:
        history = tree.history

    state_belief = {}
    h_node = tree.traverse(history)
    for hp_state, prob in h_node.belief.get_dist().items():
        state = hp_state.state   # type: ignore
        if state not in state_belief:
            state_belief[state] = 0.0
        state_belief[state] += prob
    return state_belief


def get_other_pis_belief(tree: tree_lib.BAPOSGMCP,
                         history: Optional[H.AgentHistory] = None
                         ) -> Dict[M.AgentID, Dict[parts.PolicyID, float]]:
    """Get agent's belief over other agent policies given history.

    This returns a distribution over policies for each opponent/other agent
    in the environment.
    """
    if history is None:
        history = tree.history
    h_node = tree.traverse(history)

    pi_belief: Dict[M.AgentID, Dict[parts.PolicyID, float]] = {}
    for i in range(tree.num_agents):
        if i == tree.ego_agent:
            continue
        # pylint: disable=[protected-access]
        pi_belief[i] = {pi_id: 0.0 for pi_id in tree._other_prior[i][0]}

    for hp_state, prob in h_node.belief.get_dist().items():
        pi_state = hp_state.other_policies   # type: ignore
        for i in range(tree.num_agents):
            if i == tree.ego_agent:
                continue
            pi_id = pi_state[i]
            pi_belief[i][pi_id] += prob

    return pi_belief

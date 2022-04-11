from itertools import product
from typing import Tuple, Optional, List

from baposgmcp import parts
from baposgmcp.pbt.interaction_graph import InteractionGraph


def get_klr_policy_id(agent_id: Optional[parts.AgentID],
                      k: int,
                      is_symmetric: bool) -> str:
    """Get the policy ID string for a K-level reasoning policy."""
    if is_symmetric:
        return f"pi_{k}"
    return f"pi_{k}_{agent_id}"


def parse_klr_policy_id(policy_id: str) -> Tuple[Optional[parts.AgentID], int]:
    """Parse KLR policy ID string to get reasoning level.

    Also get optional agent ID (for non-symmetric environments)
    """
    tokens = policy_id.split("_")
    if len(tokens) == 2:
        return None, int(tokens[1])

    if len(tokens) == 3:
        return tokens[2], int(tokens[1])

    raise ValueError(f"Invalid KLR Policy ID str '{policy_id}'")


def construct_klr_interaction_graph(agent_ids: List[parts.AgentID],
                                    k_levels: int,
                                    is_symmetric: bool) -> InteractionGraph:
    """Construct a K-Level Reasoning Interaction Graph.

    Note that this function constructs the graph and edges between policy IDs,
    but the actual policies still need to be added.
    """
    igraph = InteractionGraph(is_symmetric)

    for agent_id, k in product(agent_ids, range(-1, k_levels+1)):
        policy_id = get_klr_policy_id(agent_id, k, is_symmetric)
        igraph.add_policy(agent_id, policy_id, {})

    for src_agent_id, k, dest_agent_id in product(
            agent_ids, range(0, k_levels+1), agent_ids
    ):
        if not is_symmetric and src_agent_id == dest_agent_id:
            continue

        src_policy_id = get_klr_policy_id(src_agent_id, k, is_symmetric)
        dest_policy_id = get_klr_policy_id(dest_agent_id, k-1, is_symmetric)
        igraph.add_edge(
            src_agent_id,
            src_policy_id,
            dest_agent_id,
            dest_policy_id,
            1.0
        )

    return igraph

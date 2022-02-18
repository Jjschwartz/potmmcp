import random
from typing import Union, Any, Dict, Tuple, Optional, List

import posggym.model as M

AgentID = Union[str, int, M.AgentID]
PolicyID = Union[str, int]
Policy = Any


class InteractionGraph:
    """Interaction Graph for Population Based Training """

    def __init__(self):
        self._graph: Dict[PolicyID, Dict[PolicyID: float]] = {}
        self._policies: Dict[PolicyID, Policy] = {}

    def add_policy(self, policy_id: PolicyID, policy: Policy) -> None:
        """Add a policy to the interaction graph """
        self._policies[policy_id] = policy
        self._graph[policy_id] = {}

    def add_edge(self,
                 src_policy_id: PolicyID,
                 dest_policy_id: PolicyID,
                 weight: float) -> None:
        """Add a directed edge between policies on the graph

        Updates edge weight if an edge already exists between src and dest
        policies.
        """
        assert src_policy_id in self._policies, (
            f"Source policy with ID={src_policy_id} not in graph."
        )
        assert dest_policy_id in self._policies, (
            f"Destination policy with ID={dest_policy_id} not in graph."
        )
        assert 0 <= weight, (
            f"Edge weight={weight} must non-negative."
        )
        self._graph[src_policy_id][dest_policy_id] = weight

    def update_policy(self, policy_id: PolicyID, new_policy: Policy) -> None:
        """Updates stored policy """
        assert policy_id in self._policies, (
            f"Policy with ID={policy_id} not in graph. Make sure to add the "
            "policy using the add_policy() function, before updating."
        )
        self._policies[policy_id] = new_policy

    def sample_policy(self, policy_id: PolicyID) -> Tuple[PolicyID, Policy]:
        """Sample an opponent policy from the graph for given policy_id """
        assert policy_id in self._policies, (
            f"Policy with ID={policy_id} not in graph. Make sure to add the "
            "policy using the add_policy() function, before sampling."
        )
        assert len(self._graph[policy_id]) > 0, (
            f"Not edges added from policy with ID={policy_id}. Make sure to "
            "add edges from the policy using the add_edge() function, before "
            "sampling."
        )
        other_policy_ids = list(self._graph[policy_id])
        other_policy_weights = list(self._graph[policy_id].values())

        sampled_id = random.choices(
            other_policy_ids, weights=other_policy_weights, k=1
        )[0]

        return policy_id, self._policies[sampled_id]


def get_klr_policy_id(agent_id: Optional[AgentID],
                      k: int,
                      is_symmetric: bool) -> str:
    """Get the policy ID string for a K-level reasoning policy. """
    if is_symmetric:
        return f"pi_{k}"
    return f"pi_{k}_{agent_id}"


def parse_klr_policy_id(policy_id: str) -> Tuple[Optional[AgentID], int]:
    """Parse K-Level Reasoning policy ID string to get reasoning level and
    optional agent ID (for non-symmetric environments)
    """
    tokens = policy_id.split("_")
    if len(tokens) == 2:
        return None, int(tokens[1])

    if len(tokens) == 3:
        return tokens[2], int(tokens[1])

    raise ValueError(f"Invalid KLR Policy ID str '{policy_id}'")


def construct_klr_interaction_graph(agent_ids: List[AgentID],
                                    k_levels: int,
                                    is_symmetric: bool) -> InteractionGraph:
    """Constructs a K-Level Reasoning Interaction Graph

    Note that this function constructs the graph and edges between policy IDs,
    but the actual policies still need to be added.
    """
    igraph = InteractionGraph()

    if is_symmetric:
        # Don't need to worry about different agent policies for each level
        agent_ids = [agent_ids[0]]

    for k in range(-1, k_levels+1):
        for agent_id in agent_ids:
            policy_id = get_klr_policy_id(agent_id, k, is_symmetric)
            igraph.add_policy(policy_id, {})

    for k in range(0, k_levels+1):
        for src_agent_id in agent_ids:
            src_policy_id = get_klr_policy_id(src_agent_id, k, is_symmetric)
            for dest_agent_id in agent_ids:
                if not is_symmetric and src_agent_id == dest_agent_id:
                    continue
                dest_policy_id = get_klr_policy_id(
                    dest_agent_id, k-1, is_symmetric
                )
                igraph.add_edge(src_policy_id, dest_policy_id, 1.0)

    return igraph

import os
import json
import random
from typing import Dict, Tuple, Callable

from baposgmcp.parts import PolicyID, Policy, AgentID

# A function which takes the AgentID, PolicyID, Policy Object, and export
# directory and exports a policy to a local directory
PolicyExportFn = Callable[[AgentID, PolicyID, Policy, str], None]

# A function which takes the AgentID, PolicyID and import file path and imports
# a policy from the file
PolicyImportFn = Callable[[AgentID, PolicyID, str], Policy]


IGRAPH_FILE_NAME = "igraph.json"


class InteractionGraph:
    """Interaction Graph for Population Based Training

    If symmetric policies with a given ID are shared between agents, e.g. the
    policy with 'pi_0' will be the same policy for every agent in the
    environment.

    Otherwise, each agent has a seperate policies, e.g. the policy with ID
    'pi_0' will correspond to a different policy for each agent.
    """

    # Agent ID used in case of symmetric interaction graph
    # Need to use str for correct import/export to/from json
    SYMMETRIC_ID = str(None)

    def __init__(self, symmetric: bool):
        self._symmetric = symmetric
        # maps (agent_id, policy_id, other_agent_id) -> Delta(policy_id))
        self._graph: Dict[
            AgentID, Dict[PolicyID, Dict[AgentID, Dict[PolicyID, float]]]
        ] = {}
        # maps (agent_id, policy_id) -> Policy
        self._policies: Dict[AgentID, Dict[PolicyID, Policy]] = {}

    @property
    def policies(self) -> Dict[AgentID, Dict[PolicyID, Policy]]:
        """The policies for each agent in the graph """
        return self._policies

    @property
    def graph(self) -> Dict[
            AgentID, Dict[PolicyID, Dict[AgentID, Dict[PolicyID, float]]]]:
        """The graph """
        return self._graph

    def add_policy(self,
                   agent_id: AgentID,
                   policy_id: PolicyID,
                   policy: Policy) -> None:
        """Add a policy to the interaction graph

        Note, for symmetric environments agent id is treated as None.
        """
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID

        if agent_id not in self._policies:
            self._policies[agent_id] = {}
        self._policies[agent_id][policy_id] = policy

        if agent_id not in self._graph:
            self._graph[agent_id] = {}
        self._graph[agent_id][policy_id] = {}

    def add_edge(self,
                 src_agent_id: AgentID,
                 src_policy_id: PolicyID,
                 dest_agent_id: AgentID,
                 dest_policy_id: PolicyID,
                 weight: float) -> None:
        """Add a directed edge between policies on the graph

        Updates edge weight if an edge already exists between src and dest
        policies.
        """
        if self._symmetric:
            src_agent_id = self.SYMMETRIC_ID
            dest_agent_id = self.SYMMETRIC_ID

        assert src_agent_id in self._policies, (
            f"Source agent with ID={src_agent_id} not in graph."
        )
        assert src_policy_id in self._policies[src_agent_id], (
            f"Source policy with ID={src_policy_id} not in graph."
        )
        assert dest_agent_id in self._policies, (
            f"Destination agent with ID={dest_agent_id} not in graph."
        )
        assert dest_policy_id in self._policies[dest_agent_id], (
            f"Destination policy with ID={dest_policy_id} not in graph."
        )
        assert 0 <= weight, (
            f"Edge weight={weight} must be non-negative."
        )

        if dest_agent_id not in self._graph[src_agent_id][src_policy_id]:
            self._graph[src_agent_id][src_policy_id][dest_agent_id] = {}

        dest_dist = self._graph[src_agent_id][src_policy_id][dest_agent_id]
        dest_dist[dest_policy_id] = weight

    def update_policy(self,
                      agent_id: AgentID,
                      policy_id: PolicyID,
                      new_policy: Policy) -> None:
        """Updates stored policy """
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID

        assert agent_id in self._policies, (
            f"Agent with ID={agent_id} not in graph. Make sure to add the "
            "agent using the add_policy function, before updating."
        )
        assert policy_id in self._policies[agent_id], (
            f"Policy with ID={policy_id} not in graph. Make sure to add the "
            "policy using the add_policy() function, before updating."
        )
        self._policies[agent_id][policy_id] = new_policy

    def sample_policy(self,
                      agent_id: AgentID,
                      policy_id: PolicyID,
                      other_agent_id: AgentID) -> Tuple[PolicyID, Policy]:
        """Sample an other agent policy from the graph for given policy_id """
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID
            other_agent_id = self.SYMMETRIC_ID

        assert agent_id in self._policies, (
            f"Agent with ID={agent_id} not in graph. Make sure to add the "
            "agent using the add_policy function, before sampling."
        )
        assert other_agent_id in self._policies, (
            f"Other agent with ID={agent_id} not in graph. Make sure to add "
            "the agent using the add_policy function, before sampling."
        )
        assert policy_id in self._policies[agent_id], (
            f"Policy with ID={policy_id} not in graph. Make sure to add the "
            "policy using the add_policy() function, before sampling."
        )
        assert len(self._graph[agent_id][policy_id][other_agent_id]) > 0, (
            f"No edges added from policy with ID={policy_id}. Make sure to "
            "add edges from the policy using the add_edge() function, before "
            "sampling."
        )
        other_policy_dist = self._graph[agent_id][policy_id][other_agent_id]
        other_policy_ids = list(other_policy_dist)
        other_policy_weights = list(other_policy_dist.values())

        sampled_id = random.choices(
            other_policy_ids, weights=other_policy_weights, k=1
        )[0]

        return policy_id, self._policies[other_agent_id][sampled_id]

    def sample_policies(self,
                        agent_id: AgentID,
                        policy_id: PolicyID
                        ) -> Dict[AgentID, Tuple[PolicyID, Policy]]:
        """Sample a policy for each other agent from the graph, connected
        to the given (agent_id, policy_id).
        """
        other_policies: Dict[AgentID, Tuple[PolicyID, Policy]] = {}
        for other_agent_id in self._graph[agent_id][policy_id]:
            other_policies[other_agent_id] = self.sample_policy(
                agent_id, policy_id, other_agent_id
            )
        return other_policies

    def export_graph(self, export_dir: str, policy_export_fn: PolicyExportFn):
        """Export Interaction Graph to a local directory. """
        igraph_file = os.path.join(export_dir, IGRAPH_FILE_NAME)
        with open(igraph_file, "w", encoding="utf-8") as fout:
            json.dump(self._graph, fout)

        for agent_id, policy_map in self._policies.items():
            agent_dir = os.path.join(export_dir, str(agent_id))
            os.mkdir(agent_dir)

            for policy_id, policy in policy_map.items():
                policy_dir = os.path.join(agent_dir, str(policy_id))
                os.mkdir(policy_dir)

                policy_export_fn(agent_id, policy_id, policy, policy_dir)

    def import_graph(self,
                     import_dir: str,
                     policy_import_fn: PolicyImportFn):
        """Import interaction graph from a local directory.

        Note, this assumes import directory was generated using the
        InteractionGraph.export_graph function.
        """
        igraph_file = os.path.join(import_dir, IGRAPH_FILE_NAME)
        with open(igraph_file, "r", encoding="utf-8") as fin:
            self._graph = json.load(fin)

        policies: Dict[AgentID, Dict[PolicyID, Policy]] = {}
        for agent_id, policy_map in self._graph.items():
            agent_dir = os.path.join(import_dir, str(agent_id))
            policies[agent_id] = {}

            for policy_id in policy_map:
                policy_dir = os.path.join(agent_dir, str(policy_id))
                policies[agent_id][policy_id] = policy_import_fn(
                    agent_id, policy_id, policy_dir
                )

        self._policies = policies

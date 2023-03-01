import abc
import copy
import math
import random
from itertools import product
from typing import Dict, List, Union, Set

import posggym.model as M

import potmmcp.policy as P


class PolicyPrior(abc.ABC):
    """Policy prior over a set of other agent's policies."""

    def __init__(
        self, model: M.POSGModel, ego_agent: M.AgentID, policies: P.AgentPolicyMap
    ):
        assert len(policies) == model.n_agents - 1
        assert all(i in policies for i in range(model.n_agents) if i != ego_agent)
        self.model = model
        self.num_agents = model.n_agents
        self.ego_agent = ego_agent
        self.policies = policies

    @abc.abstractmethod
    def sample_agent_policy(self, agent_id: M.AgentID) -> P.PolicyID:
        """Sample policy for a specific agent from the prior."""

    @abc.abstractmethod
    def get_prior_dist(self) -> Dict[P.PolicyState, float]:
        """Get the distribution over policy states for this prior."""

    def get_agent_policy_id_map(self) -> Dict[M.AgentID, List[P.PolicyID]]:
        """Get map from agent id to list of policy IDs for that agent."""
        return {i: list(pi_map) for i, pi_map in self.policies.items()}

    def sample_policy_state(self) -> P.PolicyState:
        """Sample policy for each agent from the prior.

        The base implementation of this function samples the policy for each
        agent independently.

        Update this function to support correlated policy state sampling (e.g.
        for sampling teams of agents.)
        """
        policy_state = []
        for i in range(self.num_agents):
            if i != self.ego_agent:
                pi_id = self.sample_agent_policy(i)
            else:
                pi_id = "-1"
            policy_state.append(pi_id)
        return tuple(policy_state)

    def get_policy_objs(
        self, policy_state: P.PolicyState
    ) -> Dict[M.AgentID, P.BasePolicy]:
        """Get policy objects given polict_state."""
        policies = {}
        for i in range(self.num_agents):
            if i == self.ego_agent:
                continue
            policies[i] = self.policies[i][policy_state[i]]
        return policies


class UniformPolicyPrior(PolicyPrior):
    """Uniform prior over other agent policies."""

    def sample_agent_policy(self, agent_id: M.AgentID) -> P.PolicyID:
        return random.choice(list(self.policies[agent_id]))

    def get_prior_dist(self) -> Dict[P.PolicyState, float]:
        all_policy_tuples = list(product(*[self.policies[i] for i in self.policies]))
        prob = 1.0 / len(all_policy_tuples)

        prior = {}
        for policy_tuple in all_policy_tuples:
            policy_state = list(policy_tuple)
            policy_state.insert(self.ego_agent, -1)
            prior[tuple(policy_state)] = prob

        return prior


class MapPolicyPrior(PolicyPrior):
    """Policy Prior using a dictionary defined policy dist for each agent."""

    def __init__(
        self,
        model: M.POSGModel,
        ego_agent: M.AgentID,
        policies: P.AgentPolicyMap,
        policy_dist_map: P.AgentPolicyDist,
    ):
        super().__init__(model, ego_agent, policies)
        self._policy_dist_map = policy_dist_map

    def sample_agent_policy(self, agent_id: M.AgentID) -> P.PolicyID:
        policy_dist = self._policy_dist_map[agent_id]
        return random.choices(
            list(policy_dist), weights=list(policy_dist.values()), k=1
        )[0]

    def get_prior_dist(self) -> Dict[P.PolicyState, float]:
        prior = {}
        for policy_tuples in product(
            *[self._policy_dist_map[i].items() for i in self._policy_dist_map]
        ):
            # each policy tuple = ((pi_0_id, prob), ..., (pi_n_id, prob))
            # with one (pi_id, prob) tuple for each non-ego agent
            policy_state = list(pt[0] for pt in policy_tuples)
            policy_state.insert(self.ego_agent, -1)
            policy_state_prob = math.prod(pt[1] for pt in policy_tuples)
            prior[tuple(policy_state)] = policy_state_prob

        return prior

    @staticmethod
    def load_posggym_agents_prior(
        model: M.POSGModel, ego_agent: M.AgentID, policy_dist_map: P.AgentPolicyDist
    ) -> "MapPolicyPrior":
        import posggym_agents

        policies = {}
        for i in range(model.n_agents):
            if i == ego_agent:
                continue
            policies[i] = {
                id: posggym_agents.make(id, model, i) for id in policy_dist_map[i]
            }
        return MapPolicyPrior(model, ego_agent, policies, policy_dist_map)


class PolicyStateMapPrior(PolicyPrior):
    """Policy prior using a dictionary mapping policy-states to prior prob."""

    def __init__(
        self,
        model: M.POSGModel,
        ego_agent: M.AgentID,
        policies: P.AgentPolicyMap,
        policy_state_map: Dict[P.PolicyState, float],
    ):
        super().__init__(model, ego_agent, policies)
        self._policy_state_map = policy_state_map
        self._policy_dist_map: Dict[M.AgentID, P.PolicyDist] = {
            i: {} for i in range(model.n_agents)
        }
        for pi_state, prob in policy_state_map.items():
            for i in range(model.n_agents):
                if i == self.ego_agent:
                    continue
                pi_id = pi_state[i]
                if pi_id not in self._policy_dist_map[i]:
                    self._policy_dist_map[i][pi_id] = 0
                self._policy_dist_map[i][pi_id] += prob

    def sample_agent_policy(self, agent_id: M.AgentID) -> P.PolicyID:
        return random.choices(
            list(self._policy_dist_map[agent_id]),
            weights=list(self._policy_dist_map[agent_id].values()),
            k=1,
        )[0]

    def get_prior_dist(self) -> Dict[P.PolicyState, float]:
        return copy.copy(self._policy_state_map)

    def sample_policy_state(self) -> P.PolicyState:
        return random.choices(
            list(self._policy_state_map),
            weights=list(self._policy_state_map.values()),
            k=1,
        )[0]

    @staticmethod
    def load_posggym_agents_prior(
        model: M.POSGModel,
        ego_agent: M.AgentID,
        policy_state_map: Dict[P.PolicyState, float],
    ) -> "PolicyStateMapPrior":
        import posggym_agents

        policy_ids: Dict[M.AgentID, Set[P.PolicyID]] = {
            i: set() for i in range(model.n_agents)
        }
        for pi_state in policy_state_map:
            for i in range(model.n_agents):
                if i == ego_agent:
                    continue
                policy_ids[i].add(pi_state[i])

        policies = {}
        for i in range(model.n_agents):
            if i == ego_agent:
                continue
            policies[i] = {
                id: posggym_agents.make(id, model, i) for id in policy_ids[i]
            }
        return PolicyStateMapPrior(model, ego_agent, policies, policy_state_map)


def load_posggym_agents_policy_prior(
    model: M.POSGModel,
    ego_agent: M.AgentID,
    policy_prior_map: Union[P.AgentPolicyDist, Dict[P.PolicyState, float]],
) -> PolicyPrior:
    """Load Policy Prior of posggym agents.

    Handles loading correct prior based on policy prior map format.

    Supports loading

    MapPolicyPrior
    PolicyStateMapPrior

    """
    if isinstance(list(policy_prior_map.values())[0], dict):
        return MapPolicyPrior.load_posggym_agents_prior(
            model, ego_agent, policy_prior_map  # type: ignore
        )
    elif isinstance(list(policy_prior_map.values())[0], float):
        return PolicyStateMapPrior.load_posggym_agents_prior(
            model, ego_agent, policy_prior_map  # type: ignore
        )
    raise ValueError(f"Unsupported policy_prior_map format '{policy_prior_map}'")

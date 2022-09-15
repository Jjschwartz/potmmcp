import abc
import math
import random
from typing import Dict, List
from itertools import product

import posggym.model as M

import baposgmcp.policy as P


class PolicyPrior(abc.ABC):
    """Policy prior over a set of other agent's policies."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: M.AgentID,
                 policies: P.AgentPolicyMap):
        assert len(policies) == model.n_agents-1
        assert all(
            i in policies for i in range(model.n_agents) if i != ego_agent
        )
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
                pi_id = -1
            policy_state.append(pi_id)
        return tuple(policy_state)

    def get_policy_objs(self,
                        policy_state: P.PolicyState
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
        all_policy_tuples = list(product(
            *[self.policies[i] for i in self.policies]
        ))
        prob = 1.0 / len(all_policy_tuples)

        prior = {}
        for policy_tuple in all_policy_tuples:
            policy_state = list(policy_tuple)
            policy_state.insert(self.ego_agent, -1)
            prior[tuple(policy_state)] = prob

        return prior


class MapPolicyPrior(PolicyPrior):
    """Policy Prior using a dictionary defined policy dist for each agent."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: M.AgentID,
                 policies: P.AgentPolicyMap,
                 policy_dist_map: P.AgentPolicyDist):
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

import abc
import random
from typing import Dict
from itertools import product

import gym

import posggym.model as M

import baposgmcp.policy as P
from baposgmcp.parts import AgentID


class MetaPolicy(abc.ABC):
    """A Meta-Policy.

    A Meta-Policy maps Policy States (i.e. the list of policies of the other
    agents) to a policy for the ego agent.
    """

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: AgentID,
                 ego_policies: P.PolicyMap):
        self.model = model
        self.ego_agent = ego_agent
        self.num_agents = model.n_agents
        self.ego_policies = ego_policies

        action_space = self.model.action_spaces[self.ego_agent]
        assert isinstance(action_space, gym.spaces.Discrete)
        self.action_space = list(range(action_space.n))

    def get_policy_obj(self, policy_id: P.PolicyID) -> P.BasePolicy:
        """Get ego agent policy with given ID."""
        return self.ego_policies[policy_id]

    def sample(self, policy_state: P.PolicyState) -> P.BasePolicy:
        """Sample policy to use given policies of other agents."""
        dist = self.get_policy_dist(policy_state)
        pi_id = random.choices(list(dist), weights=dist.values(), k=1)[0]
        return self.ego_policies[pi_id]

    @abc.abstractmethod
    def get_policy_dist(self, policy_state: P.PolicyState) -> P.PolicyDist:
        """Get distribution over ego policies given other agents policies."""

    def get_exp_policy_dist(self,
                            policy_dist: P.AgentPolicyDist) -> P.PolicyDist:
        """Get meta-distribution given distribution of other agent policies."""
        exp_dist = {pi_id: 0.0 for pi_id in self.ego_policies}
        for policy_tuples in product(
                *[policy_dist[i].items() for i in policy_dist]
        ):
            # each policy tuple = ((pi_0_id, prob), ..., (pi_n_id, prob))
            # with one tuple for each other agent
            policy_state = tuple(pt[0] for pt in policy_tuples)
            policy_state_prob = sum(pt[1] for pt in policy_tuples)

            meta_policy_dist = self.get_policy_dist(policy_state)
            for pi_id, meta_prob in meta_policy_dist.items():
                exp_dist[pi_id] += policy_state_prob * meta_prob

        # normalize
        dist_sum = sum(exp_dist.values())
        for pi_id in exp_dist:
            exp_dist[pi_id] /= dist_sum

        return exp_dist

    def get_uniform_policy_dist(self) -> P.PolicyDist:
        """Get uniform distribution over ego policies."""
        n = len(self.ego_policies)
        return {pi_id: 1/n for pi_id in self.ego_policies}

    def get_exp_action_dist(self,
                            policy_dist: P.PolicyDist,
                            hidden_states: P.PolicyHiddenStateMap
                            ) -> P.ActionDist:
        """Get the expected action distribution over ego policies."""
        assert len(policy_dist) == len(hidden_states)
        assert set(hidden_states).issubset(policy_dist)

        exp_action_dist = {a: 0.0 for a in self.action_space}
        for pi_id, pi_prob in policy_dist.items():
            pi = self.ego_policies[pi_id]
            pi_action_dist = pi.get_pi_from_hidden_state(hidden_states[pi_id])
            for a, a_prob in pi_action_dist.items():
                exp_action_dist[a] += pi_prob * a_prob

        # normalize
        prob_sum = sum(exp_action_dist.values())
        for a in exp_action_dist:
            exp_action_dist[a] /= prob_sum

        return exp_action_dist


class DictMetaPolicy(MetaPolicy):
    """A Meta-Policy defined using a dictionary.

    Note, the dictionary should define an mapping from every feasable
    combination of other agent policies to distribution over ego agent
    policies.
    """

    def __init__(self,
                 ego_agent: AgentID,
                 num_agents: int,
                 ego_policies: P.PolicyMap,
                 meta_policy_dict: Dict[P.PolicyState, P.PolicyDist]):
        super().__init__(ego_agent, num_agents, ego_policies)
        self._meta_policy_dict = meta_policy_dict

    def get_policy_dist(self, policy_state: P.PolicyState) -> P.PolicyDist:
        return self._meta_policy_dict[policy_state]

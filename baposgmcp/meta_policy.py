import abc
import math
import random
from typing import Dict, List, Sequence

import gym

import posggym.model as M

import baposgmcp.policy as P


class MetaPolicy(abc.ABC):
    """A Meta-Policy.

    A Meta-Policy maps Policy States (i.e. the list of policies of the other
    agents) to a policy for the ego agent.
    """

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: M.AgentID,
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
        pi_id = random.choices(list(dist), weights=list(dist.values()), k=1)[0]
        return self.ego_policies[pi_id]

    @abc.abstractmethod
    def get_policy_dist(self, policy_state: P.PolicyState) -> P.PolicyDist:
        """Get distribution over ego policies given other agents policies."""

    def get_exp_policy_dist(self,
                            policy_state_dist: Dict[P.PolicyState, float]
                            ) -> P.PolicyDist:
        """Get meta-distribution given distribution of other agent policies."""
        exp_dist = {pi_id: 0.0 for pi_id in self.ego_policies}
        for policy_state, prob in policy_state_dist.items():
            meta_policy_dist = self.get_policy_dist(policy_state)
            for pi_id, meta_prob in meta_policy_dist.items():
                exp_dist[pi_id] += prob * meta_prob

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


class SingleMetaPolicy(MetaPolicy):
    """A Meta-Policy with only a single policy."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: M.AgentID,
                 ego_policies: P.PolicyMap):
        super().__init__(model, ego_agent, ego_policies)
        assert len(ego_policies) == 0
        self._policy_id = list(ego_policies.keys())[0]
        self._policy = ego_policies[self._policy_id]

    def get_policy_dist(self, policy_state: P.PolicyState) -> P.PolicyDist:
        return {self._policy_id: 1.0}


class UniformMetaPolicy(MetaPolicy):
    """A Meta-Policy which selects policies uniformly."""

    def get_policy_dist(self, policy_state: P.PolicyState) -> P.PolicyDist:
        return self.get_uniform_policy_dist()


class DictMetaPolicy(MetaPolicy):
    """A Meta-Policy defined using a dictionary.

    Note, the dictionary should define an mapping from every feasable
    combination of other agent policies to distribution over ego agent
    policies.
    """

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: M.AgentID,
                 ego_policies: P.PolicyMap,
                 meta_policy_dict: Dict[P.PolicyState, P.PolicyDist]):
        super().__init__(model, ego_agent, ego_policies)
        self._meta_policy_dict = meta_policy_dict

    def get_policy_dist(self, policy_state: P.PolicyState) -> P.PolicyDist:
        meta_policy = self._meta_policy_dict[policy_state]
        for pi_id in self.ego_policies:
            if pi_id not in meta_policy:
                meta_policy[pi_id] = 0.0
        return meta_policy

    @staticmethod
    def load_possgym_agents_meta_policy(model: M.POSGModel,
                                        agent_id: M.AgentID,
                                        meta_policy_dict: [
                                            Dict[P.PolicyState, P.PolicyDist]
                                        ]
                                        ) -> MetaPolicy:
        import posggym_agents

        policy_ids = set()
        for policy_dist in meta_policy_dict.values():
            policy_ids.update(policy_dist)

        policies = {
            id: posggym_agents.make(id, model, agent_id) for id in policy_ids
        }
        return DictMetaPolicy(
            model, agent_id, policies, meta_policy_dict
        )


def get_greedy_policy_dict(pairwise_returns: Dict[
                               P.PolicyState, Dict[P.PolicyID, float]]
                           ) -> [Dict[P.PolicyState, P.PolicyDist]]:
    """Get Greedy Meta-Policy Dict from pairwise returns."""
    greedy_policy = {}
    for pi_state, policy_returns in pairwise_returns.items():
        max_policies = []
        max_return = -float('inf')
        for pi_id, ret in policy_returns.items():
            if ret > max_return:
                max_return = ret
                max_policies = [pi_id]
            elif ret == max_return:
                max_policies.append(pi_id)

        greedy_policy[pi_state] = {
            pi_id: 1.0/len(max_policies) for pi_id in max_policies
        }
    return greedy_policy


def softmax(values: Sequence[float], temperature: float) -> List[float]:
    """Get softmax of array of values."""
    sum_e = sum(math.e**(z/temperature) for z in values)
    return [(math.e**(z/temperature)) / sum_e for z in values]


def get_softmax_policy_dict(pairwise_returns: Dict[
                                P.PolicyState, Dict[P.PolicyID, float]],
                            temperature: float
                            ) -> [Dict[P.PolicyState, P.PolicyDist]]:
    """Get Softmax Meta-Policy Dict from pairwise returns."""
    softmax_policy = {}
    for pi_state, policy_returns in pairwise_returns.items():
        softmax_dist = softmax(policy_returns.values(), temperature)
        softmax_policy[pi_state] = {
            pi_id: p for pi_id, p in zip(policy_returns, softmax_dist)
        }
    return softmax_policy


def get_uniform_policy_dict(pairwise_returns: Dict[
                                P.PolicyState, Dict[P.PolicyID, float]],
                            ) -> [Dict[P.PolicyState, P.PolicyDist]]:
    """Get Uniform Meta-Policy Dict from pairwise returns."""
    uniform_policy = {}
    for pi_state, policy_returns in pairwise_returns.items():
        uniform_policy[pi_state] = {
            pi_id: 1.0 / len(policy_returns) for pi_id in policy_returns
        }
    return uniform_policy

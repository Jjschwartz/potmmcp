import copy
from itertools import product
from typing import Tuple, List, Dict, Union

import posggym.model as M

import baposgmcp.policy as P
from baposgmcp.run.exp import PolicyParams
from baposgmcp.meta_policy import DictMetaPolicy
from baposgmcp.policy_prior import load_posggym_agents_policy_prior

from baposgmcp.tree.node import ObsNode
from baposgmcp.tree.policy import BAPOSGMCP
from baposgmcp.tree.hps import HistoryPolicyState


def load_pometarollout_params(variable_params: Dict[str, List],
                              baposgmcp_kwargs: Dict,
                              policy_prior_map: Union[
                                  P.AgentPolicyDist,
                                  Dict[P.PolicyState, float]
                              ],
                              meta_policy_dict: Dict[
                                  P.PolicyState, P.PolicyDist
                              ],
                              base_policy_id: str = "POMetaRollout"
                              ) -> List[PolicyParams]:
    """Load list of policy params for POMetaRollout.

    Returns policy params for pairwise combinations of num sims and
    action selection.

    """
    base_kwargs = dict(baposgmcp_kwargs)

    base_kwargs.update({
        "policy_prior_map": policy_prior_map,
        "meta_policy_dict": meta_policy_dict
    })

    policy_params = []
    param_names = list(variable_params)
    param_value_lists = [variable_params[k] for k in param_names]
    for values in product(*param_value_lists):
        # need to do copy as kwargs is modified in baposgmcp init fn
        kwargs = copy.deepcopy(base_kwargs)
        policy_id = base_policy_id
        for k, v in zip(param_names, values):
            kwargs[k] = v
            # remove _ so it's easier to parse when doing analysis
            policy_id += f"_{k.replace('_', '')}{v}"
        kwargs["policy_id"] = policy_id

        params = PolicyParams(
            id=policy_id,
            entry_point=POMetaRollout.posggym_agents_entry_point,
            kwargs=kwargs
        )
        policy_params.append(params)

    return policy_params


class POMetaRollout(BAPOSGMCP):
    """Partially Observable Meta Rollout algorithm.

    Uses Monte-Carlo Belief state updates same as BAPOSGMCP.
    It runs N rollouts from the root node using the meta-policy for rollouts
    and PUCB for selecting the actions to explore from the root node.

    It's essentially BAPOSGMCP with a single step lookahead.

    Different action selection strategies can be used via the
    `action_selection` parameter in the __init__ method.

    `pucb` `ucb` `uniform`

    """

    def _simulate(self,
                  hp_state: HistoryPolicyState,
                  obs_node: ObsNode,
                  depth: int,
                  rollout_policy: P.BasePolicy) -> Tuple[float, int]:
        # main difference is to evaluate after a single action selection
        if len(obs_node.children) == 0 or depth == 1:
            agent_history = hp_state.history.get_agent_history(self.agent_id)
            self._expand(obs_node, agent_history)
            leaf_node_value = self._evaluate(
                hp_state,
                depth,
                rollout_policy,
                obs_node.rollout_hidden_states[rollout_policy.policy_id]
            )
            return leaf_node_value, depth

        ego_action = self._search_action_selection(obs_node)
        joint_action = self._get_joint_action(hp_state, ego_action)

        joint_step = self.model.step(hp_state.state, joint_action)
        joint_obs = joint_step.observations

        ego_obs = joint_obs[self.agent_id]
        ego_return = joint_step.rewards[self.agent_id]

        new_history = hp_state.history.extend(joint_action, joint_obs)
        next_pi_state = hp_state.policy_state
        next_pi_hidden_states = self._update_other_policies(
            next_pi_state,
            hp_state.hidden_states,
            joint_action,
            joint_obs
        )
        next_hp_state = HistoryPolicyState(
            joint_step.state, new_history, next_pi_state, next_pi_hidden_states
        )

        action_node = obs_node.get_child(ego_action)
        if action_node.has_child(ego_obs):
            child_obs_node = action_node.get_child(ego_obs)
            self._update_obs_node(child_obs_node, rollout_policy)
            if not joint_step.dones[self.agent_id]:
                child_obs_node.is_absorbing = False
        else:
            child_obs_node = self._add_obs_node(
                action_node,
                ego_obs,
                {rollout_policy.policy_id: 1.0},
                init_visits=1
            )
            child_obs_node.is_absorbing = joint_step.dones[self.agent_id]
        child_obs_node.belief.add_particle(next_hp_state)

        max_depth = depth
        if not joint_step.dones[self.agent_id]:
            future_return, max_depth = self._simulate(
                next_hp_state,
                child_obs_node,
                depth+1,
                rollout_policy
            )
            ego_return += self.discount * future_return

        action_node.update(ego_return)
        return ego_return, max_depth

    @staticmethod
    def posggym_agents_entry_point(model: M.POSGModel,
                                   agent_id: M.AgentID,
                                   kwargs):
        """Initialize POMeta Policy.

        Required kwargs
        ---------------
        policy_prior_map: Union[P.AgentPolicyDist,Dict[P.PolicyState, float]]
        meta_policy_dict : Dict[P.PolicyState, P.PolicyDist]

        Plus any other arguments required by BAPOSMGPCP.__init__ (excluding
        for model, agent_id, other_policy_prior and meta_policy)

        """
        # need to do copy as kwargs is modified
        # and may be reused in a different experiment if done on the same CPU
        kwargs = copy.deepcopy(kwargs)

        other_policy_prior = load_posggym_agents_policy_prior(
            model,
            agent_id,
            policy_prior_map=kwargs.pop("policy_prior_map")
        )

        meta_policy = DictMetaPolicy.load_possgym_agents_meta_policy(
            model,
            agent_id,
            meta_policy_dict=kwargs.pop("meta_policy_dict")
        )

        return POMetaRollout(
            model,
            agent_id,
            other_policy_prior=other_policy_prior,
            meta_policy=meta_policy,
            **kwargs
        )

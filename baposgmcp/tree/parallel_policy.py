"""The Bayesian POSGMCP class."""
import math
import time
import threading
from typing import Optional, Dict, Any, Tuple

import posggym.model as M

from baposgmcp import parts
import baposgmcp.hps as H
import baposgmcp.policy as policy_lib
from baposgmcp.rllib import RllibPolicy

import baposgmcp.tree.belief as B
import baposgmcp.tree.reinvigorate as reinvig_lib
from baposgmcp.tree.node import ObsNode, ActionNode
from baposgmcp.tree.policy import (
    BAPOSGMCP, OtherAgentPolicyMap, OtherAgentPolicyDist
)

# TODO find good value for virtual loss


class ParallelBAPOSGMCP(BAPOSGMCP):
    """Bayes Adaptive POSGMCP using parallel simulations."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 other_policies: OtherAgentPolicyMap,
                 other_policy_prior: Optional[OtherAgentPolicyDist],
                 num_sims: int,
                 rollout_policy: policy_lib.BasePolicy,
                 c_init: float,
                 c_base: float,
                 truncated: bool,
                 reinvigorator: reinvig_lib.BeliefReinvigorator,
                 extra_particles_prop: float = 1.0 / 16,
                 step_limit: Optional[int] = None,
                 epsilon: float = 0.01,
                 num_parallel_sims: int = 8,
                 virtual_loss: float = 1.0,
                 **kwargs):
        super().__init__(
            model,
            ego_agent,
            gamma,
            other_policies=other_policies,
            other_policy_prior=other_policy_prior,
            num_sims=num_sims,
            rollout_policy=rollout_policy,
            c_init=c_init,
            c_base=c_base,
            truncated=truncated,
            reinvigorator=reinvigorator,
            extra_particles_prop=extra_particles_prop,
            step_limit=step_limit,
            epsilon=epsilon,
            **kwargs
        )
        self._num_parallel_sims = num_parallel_sims
        self._virtual_loss = virtual_loss

        if self.num_sims % self._num_parallel_sims != 0:
            diff = self._num_parallel_sims
            diff -= (self.num_sims % self._num_parallel_sims)
            self.num_sims += diff
            self._log_info1(
                f"{num_sims=} is not a multiple of {num_parallel_sims=}. "
                f"Num sims adjusted accordingly to {self._num_sims}"
            )

        # Threading for handling simoultaneeous simulations

    #######################################################
    # SEARCH
    #######################################################

    def get_action(self) -> M.Action:
        self._log_info1(f"Searching for num_sims={self.num_sims}")
        start_time = time.time()

        root = self.traverse(self.history)
        if len(root.children) == 0:
            self._expand(root, self.history)

        root_b = root.belief
        for _ in range(self.num_sims // self._num_parallel_sims):
            hp_states = root_b.sample_k_correlated(self._num_parallel_sims)
            # TODO run threads
            root.n += self._num_parallel_sims

        search_time = time.time() - start_time
        search_time_per_sim = search_time / self.num_sims
        self._statistics["search_time"] += search_time
        self._log_info1(f"{search_time=:.2f} s, {search_time_per_sim=:.5f}")

        return self.get_action_by_history(self.history)

    def _select(self,
                hp_state: H.HistoryPolicyState,
                obs_node: ObsNode,
                depth: int) -> Tuple[H.HistoryPolicyState, ObsNode, int]:
        if self._search_depth_limit_reached(depth):
            return hp_state, obs_node, depth

        if len(obs_node.children) < len(self.action_space):
            # Either lead node reached or
            # obs node currently being expanded by another thread
            return hp_state, obs_node, depth

        joint_action = self._get_joint_sim_action(hp_state, obs_node)
        joint_step = self.model.step(hp_state.state, joint_action)
        joint_obs = joint_step.observations

        ego_action = joint_action[self.ego_agent]
        ego_obs = joint_obs[self.ego_agent]
        # ego_return = joint_step.rewards[self.ego_agent]

        new_history = hp_state.history.extend(joint_action, joint_obs)
        next_pi_state = hp_state.other_policies

        # TODO do stateless policy updates
        # E.g. h_t = self._update_policies(h_tm1, a, o, policies)
        self._update_policies(joint_action, joint_obs, next_pi_state)
        next_pi_hidden_states = self._get_policy_hidden_states(next_pi_state)
        next_hp_state = H.HistoryPolicyState(
            joint_step.state, new_history, next_pi_state, next_pi_hidden_states
        )

        action_node = obs_node.get_child(ego_action)
        if action_node.has_child(ego_obs):
            child_obs_node = action_node.get_child(ego_obs)
        else:
            child_obs_node = self._add_obs_node(action_node, ego_obs)

        # TODO think about how/when to add particles to belief nodes
        # My guess is now
        child_obs_node.belief.add_particle(next_hp_state)

        # TODO figure out what to do with return
        # Add it to obs node to track as an extra statistic?
        # Return sequence?

        # TODO Add virtual loss

        if joint_step.done:
            return next_hp_state, child_obs_node, depth+1

        return self._select(next_hp_state, child_obs_node, depth+1)

"""The Bayesian POSGMCP class."""
import math
import time
import random
from typing import Optional, Dict, Tuple

import gym
import posggym.model as M

from baposgmcp import parts
import baposgmcp.hps as H
import baposgmcp.policy as policy_lib
from baposgmcp.rllib import RllibPolicy

import baposgmcp.tree.belief as B
import baposgmcp.tree.reinvigorate as reinvig_lib
from baposgmcp.tree.node import ObsNode, ActionNode


AgentPolicyMap = Dict[
    M.AgentID, Dict[parts.PolicyID, policy_lib.BasePolicy]
]
AgentPolicyDist = Dict[
    M.AgentID, Tuple[Tuple[parts.PolicyID, ...], Tuple[float, ...]]
]
# Map rollout policy ID to the rollout policy
RolloutPolicyMap = Dict[parts.PolicyID, policy_lib.BasePolicy]
# Map from the policy of the other agent to the rollout policy ID
RolloutPolicySelectionMap = Dict[parts.PolicyID, parts.PolicyID]
RolloutHiddenStateMap = Dict[parts.PolicyID, H.PolicyHiddenStates]


class BAPOSGMCP(policy_lib.BasePolicy):
    """Bayes Adaptive POSGMCP."""

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 num_sims: int,
                 other_policies: AgentPolicyMap,
                 other_policy_prior: Optional[AgentPolicyDist],
                 rollout_policies: RolloutPolicyMap,
                 rollout_selection: RolloutPolicySelectionMap,
                 c_init: float,
                 c_base: float,
                 truncated: bool,
                 reinvigorator: reinvig_lib.BeliefReinvigorator,
                 extra_particles_prop: float = 1.0 / 16,
                 step_limit: Optional[int] = None,
                 epsilon: float = 0.01,
                 **kwargs):
        super().__init__(model, ego_agent, gamma, **kwargs)
        assert model.n_agents == 2, "Currently only supporting 2 agents"
        assert len(other_policies) == model.n_agents-1
        assert all(
            i in other_policies
            for i in range(self.model.n_agents) if i != self.ego_agent
        )
        self._other_policies = other_policies
        # used when creating beliefs for new nodes
        self._other_policies_id_map = {
            i: list(pi_map) for i, pi_map in other_policies.items()
        }

        if other_policy_prior is None:
            self._other_prior = self._construct_uniform_prior(other_policies)
        else:
            self._other_prior = other_policy_prior

        self.num_agents = model.n_agents
        self.num_sims = num_sims

        assert isinstance(model.action_spaces[ego_agent], gym.spaces.Discrete)
        num_actions = model.action_spaces[ego_agent].n
        self.action_space = list(range(num_actions))

        self._rollout_policies = rollout_policies
        self._rollout_selection = rollout_selection
        self._c_init = c_init
        self._c_base = c_base
        self._truncated = truncated
        self._reinvigorator = reinvigorator
        self._extra_particles = math.ceil(num_sims * extra_particles_prop)
        self._step_limit = step_limit
        self._epsilon = epsilon

        if gamma == 0.0:
            self._depth_limit = 0
        else:
            self._depth_limit = math.ceil(math.log(epsilon) / math.log(gamma))

        # a belief is a dist over (state, joint history, other pi) tuples
        self._initial_belief = B.HPSParticleBelief(
            self._other_policies_id_map
        )
        self.root = ObsNode(
            None,
            None,
            self._initial_belief,
            policy={None: 1.0},
            rollout_hidden_states={
                pi_id: pi.get_initial_hidden_state()
                for pi_id, pi in self._rollout_policies.items()
            }
        )

        self._step_num = 0
        self._statistics: Dict[str, float] = {}
        self._reset_step_statistics()

    #######################################################
    # Other Policy Functions
    #######################################################

    def _construct_uniform_prior(self,
                                 other_agent_policies: AgentPolicyMap
                                 ) -> AgentPolicyDist:
        priors: AgentPolicyDist = {}
        for agent_id in range(self.model.n_agents):
            if agent_id == self.ego_agent:
                continue
            num_policies = len(other_agent_policies[agent_id])
            uniform_prob = 1.0 / num_policies
            priors[agent_id] = (
                tuple(other_agent_policies[agent_id]),
                # used cumulative prob since its faster to sample from
                # using random.choices
                tuple(uniform_prob * (i+1) for i in range(num_policies))
            )
        return priors

    def _sample_other_policy_prior(self) -> H.PolicyState:
        policy_state = []
        for i in range(self.num_agents):
            if i == self.ego_agent:
                # placeholder
                policy_id = -1
            else:
                policy_ids, probs = self._other_prior[i]
                policy_id = random.choices(policy_ids, cum_weights=probs)[0]
            policy_state.append(policy_id)
        return tuple(policy_state)

    def _get_other_policies(self,
                            policy_state: H.PolicyState
                            ) -> Dict[M.AgentID, policy_lib.BasePolicy]:
        other_policies = {}
        for i in range(self.num_agents):
            if i == self.ego_agent:
                continue
            pi = self._other_policies[i][policy_state[i]]
            other_policies[i] = pi
        return other_policies

    #######################################################
    # Step
    #######################################################

    def step(self, obs: M.Observation) -> M.Action:
        assert self._step_limit is None or self._step_num <= self._step_limit
        self._reset_step_statistics()

        self._log_info1(f"Step {self._step_num} obs={obs}")
        self.update(self._last_action, obs)

        self._last_action = self.get_action()
        self._step_num += 1
        return self._last_action

    #######################################################
    # RESET
    #######################################################

    def reset(self) -> None:
        self._log_info1("Reset")
        self._step_num = 0
        self._reset_step_statistics()
        self.history = H.AgentHistory.get_init_history()

        rollout_hidden_states = {
            pi_id: pi.get_initial_hidden_state()
            for pi_id, pi in self._rollout_policies.items()
        }
        self.root = ObsNode(
            None,
            None,
            self._initial_belief,
            policy={None: 1.0},
            rollout_hidden_states=rollout_hidden_states
        )
        self._last_action = None

        for policies in self._other_policies.values():
            for pi in policies.values():
                pi.reset()
        for policy in self._rollout_policies.values():
            policy.reset()

    def _reset_step_statistics(self):
        self._statistics = {
            "search_time": 0.0,
            "update_time": 0.0,
            "reinvigoration_time": 0.0,
            "policy_calls": 0,
            "inference_time": 0.0,
            "search_depth": 0
        }

    #######################################################
    # UPDATE
    #######################################################

    def update(self, action: M.Action, obs: M.Observation) -> None:
        self._log_info1(f"Step {self._step_num} update for a={action} o={obs}")
        start_time = time.time()
        if self.model.observation_first and self._step_num == 0:
            self.history = H.AgentHistory.get_init_history(obs)
            self._last_action = None
            self._initial_update(obs)
        else:
            self.history = self.history.extend(action, obs)
            self._update(action, obs)

        update_time = time.time() - start_time
        self._statistics["update_time"] = update_time
        self._log_info1(f"Update time = {update_time:.4f}s")

    def _initial_update(self, init_obs: M.Observation):
        action_node = self._add_action_node(self.root, None)

        init_rollout_prior = {pi_id: 0.0 for pi_id in self._rollout_policies}
        for (pi_j_id, prob) in zip(*self._other_prior[(self.ego_agent+1) % 2]):
            init_rollout_prior[self._rollout_selection[pi_j_id]] += prob

        obs_node = self._add_obs_node(
            action_node, init_obs, init_rollout_prior, 0
        )

        hps_b_0 = B.HPSParticleBelief(self._other_policies_id_map)
        b_0 = self.model.initial_belief
        while hps_b_0.size() < self.num_sims + self._extra_particles:
            # do rejection sampling from initial belief with initial obs
            state = b_0.sample()
            joint_obs = self.model.sample_initial_obs(state)

            if joint_obs[self.ego_agent] != init_obs:
                continue

            joint_history = H.JointHistory.get_init_history(
                self.num_agents, joint_obs
            )
            policy_state = self._sample_other_policy_prior()
            policy_hidden_states = self._get_initial_hidden_policy_states(
                policy_state, joint_obs
            )
            hp_state = H.HistoryPolicyState(
                state, joint_history, policy_state, policy_hidden_states
            )
            hps_b_0.add_particle(hp_state)

        obs_node.belief = hps_b_0
        self.root = obs_node
        self.root.parent = None

    def _update(self, action: M.Action, obs: M.Observation) -> None:
        self._log_debug("Pruning histories")
        # Get new root node
        try:
            a_node = self.root.get_child(action)
        except AssertionError as ex:
            if self.root.is_absorbing:
                a_node = self._add_action_node(self.root, action)
            else:
                raise ex

        try:
            obs_node = a_node.get_child(obs)

            # ensure all rollout hidden states are up-to-date in root node
            for pi_id, pi in self._rollout_policies.items():
                if pi_id in obs_node.rollout_hidden_states:
                    continue
                next_rollout_state = self._update_rollout_policy(
                    action, obs, pi, self.root.rollout_hidden_states[pi_id]
                )
                obs_node.rollout_hidden_states[pi_id] = next_rollout_state
        except AssertionError:
            # Add obs node with uniform policy prior
            # This will be updated in the course of doing simulations
            obs_node = self._add_obs_node(
                a_node, obs, None, 0
            )
            obs_node.is_absorbing = self.root.is_absorbing

        if obs_node.is_absorbing:
            self._log_debug("Absorbing state reached.")
        else:
            self._log_debug(
                f"Belief size before reinvigoration = {obs_node.belief.size()}"
            )
            self._log_debug(f"Parent belief size = {self.root.belief.size()}")
            self._reinvigorate(obs_node, action, obs)
            self._log_debug(
                f"Belief size after reinvigoration = {obs_node.belief.size()}"
            )

        self.root = obs_node
        # remove reference to parent, effectively pruning dead branches
        obs_node.parent = None

    def _get_initial_hidden_policy_states(self,
                                          policy_state: H.PolicyState,
                                          joint_obs: M.JointObservation
                                          ) -> H.PolicyHiddenStates:
        other_policies = self._get_other_policies(policy_state)
        policy_hidden_states = []
        for i in range(self.num_agents):
            a_i = None
            o_i = joint_obs[i]
            if i == self.ego_agent:
                # Don't store any hidden state for ego agent
                h_0 = None
            else:
                pi = other_policies[i]
                h_m1 = pi.get_initial_hidden_state()
                h_0 = pi.get_next_hidden_state(h_m1, a_i, o_i)
            policy_hidden_states.append(h_0)
        return tuple(policy_hidden_states)

    def _get_next_rollout_hidden_states(self,
                                        action: M.Action,
                                        obs: M.Observation,
                                        hidden_states: RolloutHiddenStateMap
                                        ) -> RolloutHiddenStateMap:
        next_hidden_states = {}
        for pi_id, pi in self._rollout_policies.items():
            next_hidden_states[pi_id] = self._update_rollout_policy(
                action, obs, pi, hidden_states[pi_id]
            )
        return next_hidden_states

    #######################################################
    # SEARCH
    #######################################################

    def get_action(self) -> M.Action:
        self._log_info1(f"Searching for num_sims={self.num_sims}")
        start_time = time.time()

        if len(self.root.children) == 0:
            self._expand(self.root, self.history)

        max_search_depth = 0
        if self.root.is_absorbing:
            self._log_debug("Agent in absorbing state. Not running search.")
        else:
            root_b = self.root.belief
            for t in range(self.num_sims):
                hp_state: H.HistoryPolicyState = root_b.sample()

                rollout_pi_id = self._select_rollout_policy(hp_state)
                rollout_pi = self._rollout_policies[rollout_pi_id]

                _, search_depth = self._simulate(
                    hp_state, self.root, 0, rollout_pi
                )
                self.root.visits += 1
                max_search_depth = max(max_search_depth, search_depth)

        search_time = time.time() - start_time
        search_time_per_sim = search_time / self.num_sims
        self._statistics["search_time"] = search_time
        self._statistics["search_depth"] = max_search_depth
        self._log_info1(
            f"{search_time=:.2f} {search_time_per_sim=:.5f} "
            f"{max_search_depth=}"
        )

        # select action with most visits
        max_actions = []
        max_visits = 0
        for a_node in self.root.children:
            if a_node.visits == max_visits:
                max_actions.append(a_node.action)
            elif a_node.visits > max_visits:
                max_visits = a_node.visits
                max_actions = [a_node.action]

        return random.choice(max_actions)

    def _simulate(self,
                  hp_state: H.HistoryPolicyState,
                  obs_node: ObsNode,
                  depth: int,
                  rollout_policy: policy_lib.BasePolicy) -> Tuple[float, int]:
        if self._search_depth_limit_reached(depth):
            return 0, depth

        if len(obs_node.children) == 0:
            # lead node reached
            agent_history = hp_state.history.get_agent_history(self.ego_agent)
            self._expand(obs_node, agent_history)
            leaf_node_value = self._evaluate(
                hp_state,
                depth,
                rollout_policy,
                obs_node.rollout_hidden_states[rollout_policy.policy_id]
            )
            return leaf_node_value, depth

        ego_action = self._get_action_from_node(obs_node, False)
        joint_action = self._get_joint_action(hp_state, ego_action)

        joint_step = self.model.step(hp_state.state, joint_action)
        joint_obs = joint_step.observations

        ego_obs = joint_obs[self.ego_agent]
        ego_return = joint_step.rewards[self.ego_agent]

        new_history = hp_state.history.extend(joint_action, joint_obs)
        next_pi_state = hp_state.other_policies
        next_pi_hidden_states = self._update_policies(
            next_pi_state,
            hp_state.hidden_states,
            joint_action,
            joint_obs
        )
        next_hp_state = H.HistoryPolicyState(
            joint_step.state, new_history, next_pi_state, next_pi_hidden_states
        )

        action_node = obs_node.get_child(ego_action)
        if action_node.has_child(ego_obs):
            child_obs_node = action_node.get_child(ego_obs)
            self._update_obs_node(child_obs_node, rollout_policy)
            if not joint_step.dones[self.ego_agent]:
                child_obs_node.is_absorbing = False
        else:
            child_obs_node = self._add_obs_node(
                action_node,
                ego_obs,
                {rollout_policy.policy_id: 1.0},
                init_visits=1
            )
            child_obs_node.is_absorbing = joint_step.dones[self.ego_agent]
        child_obs_node.belief.add_particle(next_hp_state)

        max_depth = depth
        if not joint_step.dones[self.ego_agent]:
            future_return, max_depth = self._simulate(
                next_hp_state,
                child_obs_node,
                depth+1,
                rollout_policy
            )
            ego_return += self.gamma * future_return

        action_node.update(ego_return)
        return ego_return, max_depth

    def _evaluate(self,
                  hp_state: H.HistoryPolicyState,
                  depth: int,
                  rollout_policy: policy_lib.BasePolicy,
                  rollout_state: H.PolicyHiddenState) -> float:
        if self._truncated:
            return rollout_policy.get_value_by_hidden_state(rollout_state)
        return self._rollout(hp_state, depth, rollout_policy, rollout_state)

    def _rollout(self,
                 hp_state: H.HistoryPolicyState,
                 depth: int,
                 rollout_policy: policy_lib.BasePolicy,
                 rollout_state: H.PolicyHiddenState) -> float:
        if self._search_depth_limit_reached(depth):
            return 0

        ego_action = rollout_policy.get_action_by_hidden_state(rollout_state)
        joint_action = self._get_joint_action(hp_state, ego_action)

        joint_step = self.model.step(hp_state.state, joint_action)
        joint_obs = joint_step.observations
        reward = joint_step.rewards[self.ego_agent]

        if joint_step.dones[self.ego_agent]:
            return reward

        new_history = hp_state.history.extend(joint_action, joint_obs)
        next_pi_state = hp_state.other_policies
        next_pi_hidden_states = self._update_policies(
            next_pi_state,
            hp_state.hidden_states,
            joint_action,
            joint_obs,
        )
        next_hp_state = H.HistoryPolicyState(
            joint_step.state, new_history, next_pi_state, next_pi_hidden_states
        )

        next_rollout_state = self._update_rollout_policy(
            joint_action[self.ego_agent],
            joint_obs[self.ego_agent],
            rollout_policy,
            rollout_state
        )

        future_return = self._rollout(
            next_hp_state, depth+1, rollout_policy, next_rollout_state
        )
        return reward + self.gamma * future_return

    def _search_depth_limit_reached(self, depth: int) -> bool:
        return (
            depth > self._depth_limit
            or (
                self._step_limit is not None
                and depth + self._step_num > self._step_limit
            )
        )

    def _select_rollout_policy(self,
                               hp_state: H.HistoryPolicyState
                               ) -> parts.PolicyID:
        j_id = (self.ego_agent + 1) % 2
        pi_j_id = hp_state.other_policies[j_id]
        return self._rollout_selection[pi_j_id]

    def _update_rollout_policy(self,
                               action: M.Action,
                               obs: M.Observation,
                               rollout_policy: policy_lib.BasePolicy,
                               rollout_state: H.PolicyHiddenState
                               ) -> H.PolicyHiddenState:
        start_time = time.time()
        next_rollout_state = rollout_policy.get_next_hidden_state(
            rollout_state, action, obs
        )
        self._statistics["inference_time"] += time.time() - start_time

        if isinstance(rollout_policy, RllibPolicy):
            self._statistics["policy_calls"] += 1

        return next_rollout_state

    def _update_policies(self,
                         pi_state: H.PolicyState,
                         hidden_pi_states: H.PolicyHiddenStates,
                         joint_action: M.JointAction,
                         joint_obs: M.JointObservation
                         ) -> H.PolicyHiddenStates:
        other_policies = self._get_other_policies(pi_state)
        next_hidden_policy_states = []
        for i in range(self.num_agents):
            a_i = joint_action[i]
            o_i = joint_obs[i]

            if i == self.ego_agent:
                h_t = None
            else:
                pi = other_policies[i]
                h_tm1 = hidden_pi_states[i]

                start_time = time.time()
                h_t = pi.get_next_hidden_state(h_tm1, a_i, o_i)
                self._statistics["inference_time"] += time.time() - start_time

                if isinstance(pi, RllibPolicy):
                    self._statistics["policy_calls"] += 1

            next_hidden_policy_states.append(h_t)
        return tuple(next_hidden_policy_states)

    #######################################################
    # ACTION SELECTION
    #######################################################

    def _get_action_from_node(self,
                              obs_node: ObsNode,
                              greedy: bool = False) -> M.Action:
        """Get action from given node in policy tree.

        If greedy then selects action with highest value
        else uses PUCT action selection.
        """
        if obs_node.visits == 0:
            return parts.sample_action_dist(obs_node.policy)

        sqrt_n = math.sqrt(obs_node.visits)
        exploration_rate = self._c_init
        exploration_rate += math.log(
            (1 + obs_node.visits + self._c_base) / self._c_base
        )

        max_v = -float('inf')
        max_action = obs_node.children[0].action
        for action_node in obs_node.children:
            action_v = action_node.value
            if not greedy:
                if action_node.visits == 0:
                    return action_node.action
                # add exploration bonus based on visit count and policy prior
                action_v += (
                    # set min prob > 0 so all actions have a chance to be
                    # visited during search
                    max(obs_node.policy[action_node.action], 0.01)
                    * (sqrt_n / (1 + action_node.visits))
                    * exploration_rate
                )
            if action_v > max_v:
                max_v = action_v
                max_action = action_node.action
        return max_action

    def _get_joint_action(self,
                          hp_state: H.HistoryPolicyState,
                          ego_action: M.Action) -> M.JointAction:
        agent_actions = []
        other_policies = self._get_other_policies(hp_state.other_policies)
        for i in range(self.num_agents):
            if i == self.ego_agent:
                a_i = ego_action
            else:
                pi = other_policies[i]
                a_i = pi.get_action_by_hidden_state(hp_state.hidden_states[i])
            agent_actions.append(a_i)
        return tuple(agent_actions)

    def get_pi(self) -> parts.ActionDist:
        """Get agent's distribution over actions at root of tree.

        Returns the softmax distribution over actions with temperature=1.0.
        This is used as it incorporates uncertainty based on visit counts for
        a given history.
        """
        if self.root.n == 0 or len(self.root.children) == 0:
            # uniform policy
            num_actions = len(self.action_space)
            pi = {a: 1.0 / num_actions for a in self.action_space}
            return pi

        obs_n_sqrt = math.sqrt(self.root.n)
        temp = 1.0
        pi = {
            a_node.h: math.exp(a_node.n**temp / obs_n_sqrt)
            for a_node in self.root.children
        }

        a_probs_sum = sum(pi.values())
        for a in self.action_space:
            if a not in pi:
                pi[a] = 0.0
            pi[a] /= a_probs_sum

        return pi

    def get_other_agent_pis(self,
                            hp_state: H.HistoryPolicyState
                            ) -> Dict[M.AgentID, parts.ActionDist]:
        """Get other agent policies for given history policy state."""
        other_policies = self._get_other_policies(hp_state.other_policies)
        action_dists = {}
        for i in range(self.num_agents):
            if i == self.ego_agent:
                continue
            pi = other_policies[i]
            dist = pi.get_pi_from_hidden_state(hp_state.hidden_states[i])
            action_dists[i] = dist
        return action_dists

    def get_pi_from_hidden_state(self,
                                 hidden_state: H.PolicyHiddenState
                                 ) -> parts.ActionDist:
        raise NotImplementedError()

    def get_action_by_history(self, history: H.AgentHistory) -> M.Action:
        raise NotImplementedError()

    def get_action_by_hidden_state(self,
                                   hidden_state: H.PolicyHiddenState
                                   ) -> M.Action:
        raise NotImplementedError()

    #######################################################
    # GENERAL METHODS
    #######################################################

    def get_value(self, history: Optional[H.AgentHistory]) -> float:
        assert history is None
        return self.root.value

    def get_value_by_hidden_state(self,
                                  hidden_state: H.PolicyHiddenState) -> float:
        raise NotImplementedError()

    def _expand(self, obs_node: ObsNode, history: H.AgentHistory):
        for action in self.action_space:
            if obs_node.has_child(action):
                continue
            self._add_action_node(obs_node, action)

    def _add_obs_node(self,
                      parent: ActionNode,
                      obs: M.Observation,
                      rollout_prior: Optional[Dict[parts.PolicyID, float]],
                      init_visits: int = 0
                      ) -> ObsNode:
        use_uniform_prior = rollout_prior is None
        if rollout_prior is None:
            rollout_prior = {
                pi_id: 1.0 / len(self._rollout_policies)
                for pi_id in self._rollout_policies
            }

        next_rollout_states = {}
        for pi_id in rollout_prior:
            next_rollout_states[pi_id] = self._update_rollout_policy(
                parent.action,
                obs,
                self._rollout_policies[pi_id],
                parent.parent.rollout_hidden_states[pi_id]
            )

        if use_uniform_prior:
            policy = {
                a: 1.0 / len(self.action_space) for a in self.action_space
            }
        else:
            policy = {a: 0.0 for a in self.action_space}
            for pi_id, pi_prob in rollout_prior.items():
                pi = self._rollout_policies[pi_id]
                pi_dist = pi.get_pi_from_hidden_state(
                    next_rollout_states[pi_id]
                )
                for a, a_prob in pi_dist.items():
                    policy[a] += pi_prob * a_prob

            # normalize
            prob_sum = sum(policy.values())
            for a in policy:
                policy[a] /= prob_sum

        obs_node = ObsNode(
            parent,
            obs,
            B.HPSParticleBelief(self._other_policies_id_map),
            policy=policy,
            rollout_hidden_states=next_rollout_states,
            init_value=0.0,
            init_visits=0
        )
        parent.children.append(obs_node)
        return obs_node

    def _update_obs_node(self,
                         obs_node: ObsNode,
                         rollout_policy: policy_lib.BasePolicy):
        obs_node.visits += 1

        pi_id = rollout_policy.policy_id
        if pi_id not in obs_node.rollout_hidden_states:
            next_rollout_state = self._update_rollout_policy(
                obs_node.parent.action,
                obs_node.obs,
                rollout_policy,
                obs_node.parent.parent.rollout_hidden_states[pi_id]
            )
            obs_node.rollout_hidden_states[pi_id] = next_rollout_state

        # Add rollout policy distribution to moving average policy of node
        pi_dist = rollout_policy.get_pi_from_hidden_state(
            obs_node.rollout_hidden_states[pi_id]
        )
        for a, a_prob in pi_dist.items():
            old_a_prob = obs_node.policy[a]
            obs_node.policy[a] += (a_prob - old_a_prob) / obs_node.visits

    def _add_action_node(self,
                         parent: ObsNode,
                         action: M.Action) -> ActionNode:
        action_node = ActionNode(
            parent,
            action,
            prob=parent.policy[action],
            init_value=0.0,
            init_visits=0,
            init_total_value=0.0
        )
        parent.children.append(action_node)
        return action_node

    #######################################################
    # BELIEF REINVIGORATION
    #######################################################

    def _reinvigorate(self,
                      obs_node: ObsNode,
                      action: M.Action,
                      obs: M.Observation,
                      target_node_size: Optional[int] = None) -> None:
        """Reinvigoration belief associated to given history.

        The general reinvigoration process:
        1. check belief needs to be reinvigorated (e.g. it's not a root belief)
        2. Reinvigorate node using rejection sampling/custom function for fixed
           number of tries
        3. if desired number of particles not sampled using rejection sampling/
           custom function then sample remaining particles using sampling
           without rejection
        """
        self._log_debug1("Reinvigorate")
        start_time = time.time()

        belief_size = obs_node.belief.size()
        if belief_size is None:
            # root belief
            return

        if target_node_size is None:
            target_node_size = self.num_sims + self._extra_particles

        particles_to_add = target_node_size - belief_size
        if particles_to_add <= 0:
            return

        parent_obs_node = obs_node.parent.parent
        assert parent_obs_node is not None

        self._reinvigorator.reinvigorate(
            self.ego_agent,
            obs_node.belief,
            action,
            obs,
            num_particles=particles_to_add,
            parent_belief=parent_obs_node.belief,
            joint_action_fn=self._reinvigorate_action_fn,
            joint_update_fn=self._reinvigorate_update_fn,
            **{
                "use_rejected_samples": True   # used for rejection sampling
            }
        )

        reinvig_time = time.time() - start_time
        self._statistics["reinvigoration_time"] += reinvig_time
        self._log_debug1(f"Reinvigorate: time={reinvig_time:.6f}")

    def _reinvigorate_action_fn(self,
                                hp_state: H.HistoryPolicyState,
                                ego_action: M.Action) -> M.JointAction:
        return self._get_joint_action(hp_state, ego_action)

    def _reinvigorate_update_fn(self,
                                hp_state: H.HistoryPolicyState,
                                joint_action: M.JointAction,
                                joint_obs: M.JointObservation
                                ) -> H.PolicyHiddenStates:
        pi_state = hp_state.other_policies
        return self._update_policies(
            pi_state, hp_state.hidden_states, joint_action, joint_obs
        )

    #######################################################
    # Logging
    #######################################################

    def _format_msg(self, msg: str):
        return f"i={self.ego_agent} {msg}"

    def __str__(self):
        return f"{self.__class__.__name__}"

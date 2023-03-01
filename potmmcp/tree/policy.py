import math
import random
import time
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import posggym.model as M
from posggym.utils.history import AgentHistory, JointHistory
from scipy.stats import dirichlet

import potmmcp.policy as P
import potmmcp.tree.belief as B
from potmmcp.meta_policy import MetaPolicy
from potmmcp.policy_prior import PolicyPrior
from potmmcp.tree.hps import HistoryPolicyState
from potmmcp.tree.node import ActionNode, ObsNode
from potmmcp.tree.reinvigorate import BABeliefRejectionSampler, BeliefReinvigorator
from potmmcp.tree.stats import KnownBounds, MinMaxStats


class POTMMCP(P.POTMMCPBasePolicy):
    """Partially Observable Type-base Meta Monte-Carlo Planning.

    Different action selection strategies can be used via the
    `action_selection` parameter in the __init__ method.

    `pucb` `ucb` `uniform`

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: M.AgentID,
        discount: float,
        num_sims: Optional[int],
        search_time_limit: Optional[float],
        other_policy_prior: PolicyPrior,
        meta_policy: MetaPolicy,
        c: float,
        truncated: bool,
        reinvigorator: Optional[BeliefReinvigorator] = None,
        action_selection: str = "pucb",
        dirichlet_alpha: Optional[float] = None,
        root_exploration_fraction: float = 0.25,
        known_bounds: Optional[KnownBounds] = None,
        extra_particles_prop: float = 1.0 / 16,
        step_limit: Optional[int] = None,
        epsilon: float = 0.01,
        **kwargs,
    ):
        policy_id = kwargs.pop("policy_id", f"potmmcp_{action_selection}")
        super().__init__(
            model, agent_id, policy_id=policy_id, discount=discount, **kwargs
        )

        self.num_agents = model.n_agents
        self.num_sims = num_sims
        self.search_time_limit = search_time_limit

        err_msg = (
            "Must specify either num_sims or search_time_limit, not both or niether."
        )
        if num_sims is None:
            assert search_time_limit is not None, err_msg
            self._num_particles = math.ceil(100 * search_time_limit)
        elif search_time_limit is None:
            assert num_sims is not None, err_msg
            self._num_particles = num_sims
        else:
            raise AssertionError(err_msg)

        assert isinstance(model.action_spaces[agent_id], gym.spaces.Discrete)
        num_actions = model.action_spaces[agent_id].n
        self.action_space = list(range(num_actions))

        self._meta_policy = meta_policy
        self._other_policy_prior = other_policy_prior
        self._c = c
        self._known_bounds = known_bounds
        self._min_max_stats = MinMaxStats(known_bounds)
        self._truncated = truncated
        self._extra_particles = math.ceil(self._num_particles * extra_particles_prop)
        self._step_limit = step_limit
        self._epsilon = epsilon

        if reinvigorator is None:
            reinvigorator = BABeliefRejectionSampler(
                model, sample_limit=4 * self._num_particles
            )
        self._reinvigorator = reinvigorator

        if discount == 0.0:
            self._depth_limit = 0
        else:
            self._depth_limit = math.ceil(math.log(epsilon) / math.log(discount))

        if dirichlet_alpha is None:
            dirichlet_alpha = num_actions / 10
        self._dirichlet_alpha = dirichlet_alpha
        self._root_exploration_fraction = root_exploration_fraction
        # compute once and reuse
        self._mean_exploration_noise = dirichlet(
            [self._dirichlet_alpha] * len(self.action_space)
        ).mean()

        action_selection = action_selection.lower()
        self._action_selection_mode = action_selection
        if action_selection == "pucb":
            self._search_action_selection = self.pucb_action_selection
            self._final_action_selection = self.max_visit_action_selection
        elif action_selection == "ucb":
            self._search_action_selection = self.ucb_action_selection
            self._final_action_selection = self.max_value_action_selection
        elif action_selection == "uniform":
            self._search_action_selection = self.min_visit_action_selection
            self._final_action_selection = self.max_value_action_selection
        else:
            raise ValueError(f"Invalid action selection mode '{action_selection}'")
        self._log_info1(f"Using {action_selection=}")

        # a belief is a dist over (state, joint history, other pi) tuples
        self._initial_belief = self._init_belief()
        self.root = self._init_root(self._initial_belief)

        self._step_num = 0
        self._statistics: Dict[str, float] = {}
        self._reset_step_statistics()

    def _init_belief(self) -> B.HPSParticleBelief:
        initial_belief = B.HPSParticleBelief(
            self._other_policy_prior.get_agent_policy_id_map()
        )
        if self.model.observation_first:
            # initial belief populated in _initial_update after first obs
            return initial_belief

        state_particles = self.model.initial_belief.sample_k(
            self._num_particles + self._extra_particles
        )
        for s in state_particles:
            joint_history = JointHistory.get_init_history(self.num_agents)
            policy_state = self._other_policy_prior.sample_policy_state()
            policy_hidden_states = self._get_initial_hidden_policy_states(
                policy_state, None
            )
            hp_state = HistoryPolicyState(
                s, joint_history, policy_state, policy_hidden_states
            )

            initial_belief.add_particle(hp_state)

        return initial_belief

    def _init_root(self, initial_belief: B.HPSParticleBelief) -> ObsNode:
        hidden_states = {
            pi_id: pi.get_initial_hidden_state()
            for pi_id, pi in self._meta_policy.ego_policies.items()
        }

        if self.model.observation_first:
            policy = {None: 1.0}
        else:
            meta_prior = self._meta_policy.get_exp_policy_dist(
                self._other_policy_prior.get_prior_dist()
            )
            policy = self._meta_policy.get_exp_action_dist(meta_prior, hidden_states)

        return ObsNode(
            None,
            None,
            initial_belief,
            policy=policy,
            rollout_hidden_states=hidden_states,
        )

    #######################################################
    # Step
    #######################################################

    def step(self, obs: M.Observation) -> M.Action:
        assert self._step_limit is None or self._step_num <= self._step_limit
        if self.root.is_absorbing:
            for k in self._statistics:
                self._statistics[k] = np.nan
            return self._last_action

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
        self._min_max_stats = MinMaxStats(self._known_bounds)
        self._reset_step_statistics()
        self.history = AgentHistory.get_init_history()
        self._last_action = None
        self._initial_belief = self._init_belief()
        self.root = self._init_root(self._initial_belief)

        for i, agent_policies in self._other_policy_prior.policies.items():
            for pi in agent_policies.values():
                pi.reset()

        for policy in self._meta_policy.ego_policies.values():
            policy.reset()

    def _reset_step_statistics(self):
        self._statistics = {
            "search_time": 0.0,
            "update_time": 0.0,
            "reinvigoration_time": 0.0,
            "evaluation_time": 0.0,
            "policy_calls": 0,
            "inference_time": 0.0,
            "search_depth": 0,
            "num_sims": 0,
            "min_value": self._min_max_stats.minimum,
            "max_value": self._min_max_stats.maximum,
        }

    #######################################################
    # UPDATE
    #######################################################

    def update(self, action: M.Action, obs: M.Observation) -> None:
        self._log_info1(f"Step {self._step_num} update for a={action} o={obs}")
        if self.root.is_absorbing:
            return

        start_time = time.time()
        if self.model.observation_first and self._step_num == 0:
            self.history = AgentHistory.get_init_history(obs)
            self._last_action = None
            self._initial_update(obs)
        elif self._step_num > 0:
            self.history = self.history.extend(action, obs)
            self._update(action, obs)

        update_time = time.time() - start_time
        self._statistics["update_time"] = update_time
        self._log_info1(f"Update time = {update_time:.4f}s")

    def _initial_update(self, init_obs: M.Observation):
        action_node = self._add_action_node(self.root, None)
        init_rollout_prior = self._meta_policy.get_exp_policy_dist(
            self._other_policy_prior.get_prior_dist()
        )
        obs_node = self._add_obs_node(action_node, init_obs, init_rollout_prior, 0)

        try:
            b_0 = self.model.get_agent_initial_belief(self.agent_id, init_obs)
            rejection_sample = False
        except NotImplementedError:
            b_0 = self.model.initial_belief
            rejection_sample = True

        hps_b_0 = B.HPSParticleBelief(
            self._other_policy_prior.get_agent_policy_id_map()
        )
        while hps_b_0.size() < self._num_particles + self._extra_particles:
            # do rejection sampling from initial belief with initial obs
            state = b_0.sample()
            joint_obs = self.model.sample_initial_obs(state)

            if rejection_sample and joint_obs[self.agent_id] != init_obs:
                continue

            joint_history = JointHistory.get_init_history(self.num_agents, joint_obs)
            policy_state = self._other_policy_prior.sample_policy_state()
            policy_hidden_states = self._get_initial_hidden_policy_states(
                policy_state, joint_obs
            )
            hp_state = HistoryPolicyState(
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
            for pi_id, pi in self._meta_policy.ego_policies.items():
                if pi_id in obs_node.rollout_hidden_states:
                    continue
                next_rollout_state = self._update_policy_hidden_state(
                    action, obs, pi, self.root.rollout_hidden_states[pi_id]
                )
                obs_node.rollout_hidden_states[pi_id] = next_rollout_state
        except AssertionError:
            # Add obs node with uniform policy prior
            # This will be updated in the course of doing simulations
            obs_node = self._add_obs_node(a_node, obs, None, 0)
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

    def _get_initial_hidden_policy_states(
        self, policy_state: P.PolicyState, joint_obs: Optional[M.JointObservation]
    ) -> P.PolicyHiddenStates:
        other_policies = self._other_policy_prior.get_policy_objs(policy_state)

        if joint_obs is None:
            joint_obs = [None] * self.num_agents

        policy_hidden_states = []
        for i in range(self.num_agents):
            a_i = None
            o_i = joint_obs[i]
            if i == self.agent_id:
                # Don't store any hidden state for ego agent
                h_0 = None
            else:
                pi = other_policies[i]
                h_m1 = pi.get_initial_hidden_state()
                if self.model.observation_first:
                    h_0 = pi.get_next_hidden_state(h_m1, a_i, o_i)
                else:
                    h_0 = h_m1
            policy_hidden_states.append(h_0)
        return tuple(policy_hidden_states)

    def _get_next_rollout_hidden_states(
        self,
        action: M.Action,
        obs: M.Observation,
        hidden_states: P.PolicyHiddenStateMap,
    ) -> P.PolicyHiddenStateMap:
        next_hidden_states = {}
        for pi_id, pi in self._meta_policy.ego_policies.items():
            next_hidden_states[pi_id] = self._update_policy_hidden_state(
                action, obs, pi, hidden_states[pi_id]
            )
        return next_hidden_states

    #######################################################
    # SEARCH
    #######################################################

    def get_action(self) -> M.Action:
        if self.num_sims is None:
            self._log_info1(f"Searching for search_time_limit={self.search_time_limit}")
        else:
            self._log_info1(f"Searching for num_sims={self.num_sims}")

        if self.root.is_absorbing:
            self._log_debug("Agent in absorbing state. Not running search.")
            return self.action_space[0]

        start_time = time.time()

        if len(self.root.children) == 0:
            self._expand(self.root, self.history)

        max_search_depth = 0
        n_sims = 0
        root_b = self.root.belief
        while (self.num_sims is not None and n_sims < self.num_sims) or (
            self.search_time_limit is not None
            and time.time() - start_time < self.search_time_limit
        ):
            hp_state: HistoryPolicyState = root_b.sample()
            rollout_pi = self._meta_policy.sample(hp_state.policy_state)
            _, search_depth = self._simulate(hp_state, self.root, 0, rollout_pi)
            self.root.visits += 1
            max_search_depth = max(max_search_depth, search_depth)
            n_sims += 1

        search_time = time.time() - start_time
        self._statistics["search_time"] = search_time
        self._statistics["search_depth"] = max_search_depth
        self._statistics["num_sims"] = n_sims
        self._statistics["min_value"] = self._min_max_stats.minimum
        self._statistics["max_value"] = self._min_max_stats.maximum
        self._log_info1(f"{search_time=:.2f} {max_search_depth=}")
        if self._known_bounds is None:
            self._log_info1(
                f"{self._min_max_stats.minimum=:.2f} "
                f"{self._min_max_stats.maximum=:.2f}"
            )
        self._log_info1(f"Root node policy prior = {self.root.policy_str()}")

        return self._final_action_selection(self.root)

    def _simulate(
        self,
        hp_state: HistoryPolicyState,
        obs_node: ObsNode,
        depth: int,
        rollout_policy: P.BasePolicy,
    ) -> Tuple[float, int]:
        if self._search_depth_limit_reached(depth):
            return 0, depth

        if len(obs_node.children) == 0:
            # lead node reached
            agent_history = hp_state.history.get_agent_history(self.agent_id)
            self._expand(obs_node, agent_history)
            leaf_node_value = self._evaluate(
                hp_state,
                depth,
                rollout_policy,
                obs_node.rollout_hidden_states[rollout_policy.policy_id],
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
            next_pi_state, hp_state.hidden_states, joint_action, joint_obs
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
                action_node, ego_obs, {rollout_policy.policy_id: 1.0}, init_visits=1
            )
            child_obs_node.is_absorbing = joint_step.dones[self.agent_id]
        child_obs_node.belief.add_particle(next_hp_state)

        max_depth = depth
        if not joint_step.dones[self.agent_id]:
            future_return, max_depth = self._simulate(
                next_hp_state, child_obs_node, depth + 1, rollout_policy
            )
            ego_return += self.discount * future_return

        action_node.update(ego_return)
        self._min_max_stats.update(action_node.value)
        return ego_return, max_depth

    def _evaluate(
        self,
        hp_state: HistoryPolicyState,
        depth: int,
        rollout_policy: P.BasePolicy,
        rollout_state: P.PolicyHiddenState,
    ) -> float:
        start_time = time.time()
        if self._truncated:
            v = rollout_policy.get_value_by_hidden_state(rollout_state)
        else:
            v = self._rollout(hp_state, depth, rollout_policy, rollout_state)
        self._statistics["evaluation_time"] += time.time() - start_time
        return v

    def _rollout(
        self,
        hp_state: HistoryPolicyState,
        depth: int,
        rollout_policy: P.BasePolicy,
        rollout_state: P.PolicyHiddenState,
    ) -> float:
        if self._search_depth_limit_reached(depth):
            return 0

        ego_action = rollout_policy.get_action_by_hidden_state(rollout_state)
        joint_action = self._get_joint_action(hp_state, ego_action)

        joint_step = self.model.step(hp_state.state, joint_action)
        joint_obs = joint_step.observations
        reward = joint_step.rewards[self.agent_id]

        if joint_step.dones[self.agent_id]:
            return reward

        new_history = hp_state.history.extend(joint_action, joint_obs)
        next_pi_state = hp_state.policy_state
        next_pi_hidden_states = self._update_other_policies(
            next_pi_state,
            hp_state.hidden_states,
            joint_action,
            joint_obs,
        )
        next_hp_state = HistoryPolicyState(
            joint_step.state, new_history, next_pi_state, next_pi_hidden_states
        )

        next_rollout_state = self._update_policy_hidden_state(
            joint_action[self.agent_id],
            joint_obs[self.agent_id],
            rollout_policy,
            rollout_state,
        )

        future_return = self._rollout(
            next_hp_state, depth + 1, rollout_policy, next_rollout_state
        )
        return reward + self.discount * future_return

    def _search_depth_limit_reached(self, depth: int) -> bool:
        return depth > self._depth_limit or (
            self._step_limit is not None and depth + self._step_num > self._step_limit
        )

    def _update_policy_hidden_state(
        self,
        action: M.Action,
        obs: M.Observation,
        policy: P.BasePolicy,
        hidden_state: P.PolicyHiddenState,
    ) -> P.PolicyHiddenState:
        start_time = time.time()
        next_hidden_state = policy.get_next_hidden_state(hidden_state, action, obs)
        self._statistics["inference_time"] += time.time() - start_time
        self._statistics["policy_calls"] += 1

        return next_hidden_state

    def _update_other_policies(
        self,
        pi_state: P.PolicyState,
        hidden_pi_states: P.PolicyHiddenStates,
        joint_action: M.JointAction,
        joint_obs: M.JointObservation,
    ) -> P.PolicyHiddenStates:
        other_policies = self._other_policy_prior.get_policy_objs(pi_state)
        next_hidden_policy_states = []
        for i in range(self.num_agents):
            a_i = joint_action[i]
            o_i = joint_obs[i]

            if i == self.agent_id:
                h_t = None
            else:
                pi = other_policies[i]
                h_tm1 = hidden_pi_states[i]
                h_t = self._update_policy_hidden_state(a_i, o_i, pi, h_tm1)
            next_hidden_policy_states.append(h_t)
        return tuple(next_hidden_policy_states)

    #######################################################
    # ACTION SELECTION
    #######################################################

    def pucb_action_selection(self, obs_node: ObsNode) -> M.Action:
        """Select action from node using PUCB."""
        if obs_node.visits == 0:
            # sample action using prior policy
            return random.choices(
                list(obs_node.policy.keys()),
                weights=list(obs_node.policy.values()),
                k=1,
            )[0]

        sqrt_n = math.sqrt(obs_node.visits)

        max_v = -float("inf")
        max_action = obs_node.children[0].action
        prior = self._get_exploration_prior(obs_node.policy)
        for action_node in obs_node.children:
            explore_v = (
                self._c
                * prior[action_node.action]
                * (sqrt_n / (1 + action_node.visits))
            )
            if action_node.visits > 0:
                action_v = self._min_max_stats.normalize(action_node.value)
            else:
                action_v = 0
            action_v = action_v + explore_v
            if action_v > max_v:
                max_v = action_v
                max_action = action_node.action
        return max_action

    def ucb_action_selection(self, obs_node: ObsNode) -> M.Action:
        """Select action from node using UCB."""
        if obs_node.visits == 0:
            return random.choice(self.action_space)

        log_n = math.log(obs_node.visits)

        max_v = -float("inf")
        max_action = obs_node.children[0].action
        for action_node in obs_node.children:
            if action_node.visits == 0:
                return action_node.action
            explore_v = self._c * math.sqrt(log_n / action_node.visits)
            action_v = self._min_max_stats.normalize(action_node.value) + explore_v
            if action_v > max_v:
                max_v = action_v
                max_action = action_node.action
        return max_action

    def min_visit_action_selection(self, obs_node: ObsNode) -> M.Action:
        """Select action from node with least visits.

        Note this guarantees all actions are visited equally +/- 1 when used
        during search.
        """
        if obs_node.visits == 0:
            return random.choice(self.action_space)

        min_n = obs_node.visits + 1
        next_action = obs_node.children[0].action
        for action_node in obs_node.children:
            if action_node.visits < min_n:
                min_n = action_node.visits
                next_action = action_node.action
        return next_action

    def max_visit_action_selection(self, obs_node: ObsNode) -> M.Action:
        """Select action from node with most visits.

        Breaks ties randomly.
        """
        if obs_node.visits == 0:
            return random.choice(self.action_space)

        max_actions = []
        max_visits = 0
        for a_node in obs_node.children:
            if a_node.visits == max_visits:
                max_actions.append(a_node.action)
            elif a_node.visits > max_visits:
                max_visits = a_node.visits
                max_actions = [a_node.action]
        return random.choice(max_actions)

    def max_value_action_selection(self, obs_node: ObsNode) -> M.Action:
        """Select action from node with maximum value.

        Breaks ties randomly.
        """
        if len(obs_node.children) == 0:
            # Node not expanded so select random action
            return random.choice(self.action_space)

        max_actions = []
        max_value = -float("inf")
        for a_node in obs_node.children:
            if a_node.value == max_value:
                max_actions.append(a_node.action)
            elif a_node.value > max_value:
                max_value = a_node.value
                max_actions = [a_node.action]
        return random.choice(max_actions)

    def _get_joint_action(
        self, hp_state: HistoryPolicyState, ego_action: M.Action
    ) -> M.JointAction:
        agent_actions = []
        pi_state, hidden_states = hp_state.policy_state, hp_state.hidden_states
        policies = self._other_policy_prior.get_policy_objs(pi_state)
        for i in range(self.num_agents):
            if i == self.agent_id:
                a_i = ego_action
            else:
                a_i = policies[i].get_action_by_hidden_state(hidden_states[i])
            agent_actions.append(a_i)
        return tuple(agent_actions)

    def get_pi(self) -> P.ActionDist:
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

    def get_other_agent_pis(
        self, hp_state: HistoryPolicyState
    ) -> Dict[M.AgentID, P.ActionDist]:
        """Get other agent policies for given history policy state."""
        pi_state, hidden_states = hp_state.policy_state, hp_state.hidden_states
        policies = self._other_policy_prior.get_policy_objs(pi_state)
        action_dists = {}
        for i in range(self.num_agents):
            if i == self.agent_id:
                continue
            dist = policies[i].get_pi_from_hidden_state(hidden_states[i])
            action_dists[i] = dist
        return action_dists

    def get_pi_from_hidden_state(
        self, hidden_state: P.PolicyHiddenState
    ) -> P.ActionDist:
        raise NotImplementedError()

    def get_action_by_history(self, history: AgentHistory) -> M.Action:
        raise NotImplementedError()

    def get_action_by_hidden_state(self, hidden_state: P.PolicyHiddenState) -> M.Action:
        raise NotImplementedError()

    #######################################################
    # GENERAL METHODS
    #######################################################

    def get_value(self, history: Optional[AgentHistory]) -> float:
        assert history is None
        return self.root.value

    def get_value_by_hidden_state(self, hidden_state: P.PolicyHiddenState) -> float:
        raise NotImplementedError()

    def _expand(self, obs_node: ObsNode, history: AgentHistory):
        for action in self.action_space:
            if obs_node.has_child(action):
                continue
            self._add_action_node(obs_node, action)

    def _get_exploration_prior(self, policy: P.ActionDist) -> P.ActionDist:
        mean_noise = self._mean_exploration_noise
        frac = self._root_exploration_fraction
        new_prior = {}
        for i, a in enumerate(self.action_space):
            new_prior[a] = policy[a] * (1 - frac) + mean_noise[i] * frac
        return new_prior

    def _add_obs_node(
        self,
        parent: ActionNode,
        obs: M.Observation,
        rollout_prior: Optional[P.PolicyDist],
        init_visits: int = 0,
    ) -> ObsNode:
        use_uniform_prior = rollout_prior is None
        if rollout_prior is None:
            rollout_prior = self._meta_policy.get_uniform_policy_dist()

        next_rollout_states = {}
        for pi_id in rollout_prior:
            next_rollout_states[pi_id] = self._update_policy_hidden_state(
                parent.action,
                obs,
                self._meta_policy.get_policy_obj(pi_id),
                parent.parent.rollout_hidden_states[pi_id],
            )

        if use_uniform_prior:
            policy = {a: 1.0 / len(self.action_space) for a in self.action_space}
        else:
            policy = self._meta_policy.get_exp_action_dist(
                rollout_prior, next_rollout_states
            )

        obs_node = ObsNode(
            parent,
            obs,
            B.HPSParticleBelief(self._other_policy_prior.get_agent_policy_id_map()),
            policy=policy,
            rollout_hidden_states=next_rollout_states,
            init_value=0.0,
            init_visits=init_visits,
        )
        parent.children.append(obs_node)

        return obs_node

    def _update_obs_node(self, obs_node: ObsNode, policy: P.BasePolicy):
        obs_node.visits += 1

        pi_id = policy.policy_id
        if pi_id not in obs_node.rollout_hidden_states:
            next_rollout_state = self._update_policy_hidden_state(
                obs_node.parent.action,
                obs_node.obs,
                policy,
                obs_node.parent.parent.rollout_hidden_states[pi_id],
            )
            obs_node.rollout_hidden_states[pi_id] = next_rollout_state

        # Add rollout policy distribution to moving average policy of node
        pi_dist = policy.get_pi_from_hidden_state(obs_node.rollout_hidden_states[pi_id])
        for a, a_prob in pi_dist.items():
            old_a_prob = obs_node.policy[a]
            obs_node.policy[a] += (a_prob - old_a_prob) / obs_node.visits

    def _add_action_node(self, parent: ObsNode, action: M.Action) -> ActionNode:
        action_node = ActionNode(
            parent,
            action,
            prob=parent.policy[action],
            init_value=0.0,
            init_visits=0,
            init_total_value=0.0,
        )
        parent.children.append(action_node)
        return action_node

    #######################################################
    # BELIEF REINVIGORATION
    #######################################################

    def _reinvigorate(
        self,
        obs_node: ObsNode,
        action: M.Action,
        obs: M.Observation,
        target_node_size: Optional[int] = None,
    ) -> None:
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
            target_node_size = self._num_particles + self._extra_particles

        particles_to_add = target_node_size - belief_size
        if particles_to_add <= 0:
            return

        parent_obs_node = obs_node.parent.parent
        assert parent_obs_node is not None

        self._reinvigorator.reinvigorate(
            self.agent_id,
            obs_node.belief,
            action,
            obs,
            num_particles=particles_to_add,
            parent_belief=parent_obs_node.belief,
            joint_action_fn=self._reinvigorate_action_fn,
            joint_update_fn=self._reinvigorate_update_fn,
            **{"use_rejected_samples": True},  # used for rejection sampling
        )

        reinvig_time = time.time() - start_time
        self._statistics["reinvigoration_time"] += reinvig_time
        self._log_debug1(f"Reinvigorate: time={reinvig_time:.6f}")

    def _reinvigorate_action_fn(
        self, hp_state: HistoryPolicyState, ego_action: M.Action
    ) -> M.JointAction:
        return self._get_joint_action(hp_state, ego_action)

    def _reinvigorate_update_fn(
        self,
        hp_state: HistoryPolicyState,
        joint_action: M.JointAction,
        joint_obs: M.JointObservation,
    ) -> P.PolicyHiddenStates:
        pi_state = hp_state.policy_state
        return self._update_other_policies(
            pi_state, hp_state.hidden_states, joint_action, joint_obs
        )

    #######################################################
    # Logging
    #######################################################

    def _format_msg(self, msg: str):
        return f"i={self.agent_id} {msg}"

    def __str__(self):
        return f"{self.__class__.__name__}"

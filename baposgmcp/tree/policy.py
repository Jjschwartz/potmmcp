"""The Bayesian POSGMCP class """
import math
import time
import random
from typing import Optional, Dict

import gym
import posggym.model as M

import posgmcp.belief as B
import posgmcp.history as H
from posgmcp.tree.node import Node
import posgmcp.policy as policy_lib
import posgmcp.reinvigorate as reinvig_lib

from baposgmcp import parts
import baposgmcp.hps as HPS

OtherAgentPolicyMap = Dict[
    M.AgentID, Dict[parts.PolicyID, policy_lib.BasePolicy]
]


class BAPOSGMCP(policy_lib.BasePolicy):
    """Bayes Adaptive POSGMCP """

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 other_policies: OtherAgentPolicyMap,
                 other_policy_prior: Optional[parts.OtherAgentPolicyDist],
                 num_sims: int,
                 rollout_policy: policy_lib.BaseRolloutPolicy,
                 uct_c: float,
                 reinvigorator: reinvig_lib.BeliefReinvigorator,
                 extra_particles_prop: float = 1.0 / 16,
                 step_limit: Optional[int] = None,
                 epsilon: float = 0.01,
                 **kwargs):
        super().__init__(model, ego_agent, gamma, **kwargs)
        assert len(other_policies) == model.num_agents-1
        assert all(
            i in other_policies
            for i in range(self.model.num_agents) if i != self.ego_agent
        )
        self._other_policies = other_policies

        if other_policy_prior is None:
            self._other_prior = self._construct_uniform_prior(other_policies)
        else:
            self._other_prior = other_policy_prior

        self.num_agents = model.num_agents
        self.num_sims = num_sims

        assert isinstance(model.action_spaces[ego_agent], gym.spaces.Discrete)
        num_actions = model.action_spaces[ego_agent].n
        self.action_space = list(range(num_actions))

        self._rollout_policy = rollout_policy
        self._uct_c = uct_c
        self._reinvigorator = reinvigorator
        self._extra_particles = math.ceil(num_sims * extra_particles_prop)
        self._step_limit = step_limit
        self._epsilon = epsilon

        if gamma == 0.0:
            self._depth_limit = 0
        else:
            self._depth_limit = math.ceil(math.log(epsilon) / math.log(gamma))

        # a belief is a distribution over (state, joint history, other pi)
        # tuples
        b_0 = B.ParticleBelief()
        self._initial_belief = b_0
        self.root = Node(None, None, b_0)

        self._step_num = 0
        self._statistics: Dict[str, float] = {}

    #######################################################
    # Other Policy Functions
    #######################################################

    def _construct_uniform_prior(self,
                                 other_agent_policies: OtherAgentPolicyMap
                                 ) -> parts.OtherAgentPolicyDist:
        priors: parts.OtherAgentPolicyDist = {}
        for agent_id in range(self.model.num_agents):
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

    def _sample_other_policy_prior(self) -> HPS.PolicyState:
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
                            policy_state: HPS.PolicyState
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
        self._log_debug(f"history={self.history}")

        self._last_action = self.get_action()
        self._step_num += 1
        return self._last_action

    #######################################################
    # RESET
    #######################################################

    def reset(self) -> None:
        self._log_info1("Reset")
        self._step_num = 0
        self.history = H.AgentHistory.get_init_history()
        self.root = Node(None, None, self._initial_belief)
        self._last_action = None
        for policies in self._other_policies.values():
            for pi in policies.values():
                pi.reset()
        self._reset_step_statistics()

    def _reset_step_statistics(self):
        self._statistics = {
            "search_time": 0.0,
            "update_time": 0.0,
            "reinvigoration_time": 0.0
        }

    #######################################################
    # UPDATE
    #######################################################

    def update(self, action: M.Action, obs: M.Observation) -> None:
        self._log_info1(f"Step {self._step_num} update for a={action} o={obs}")
        start_time = time.time()
        if self._step_num == 0:
            self.history = H.AgentHistory.get_init_history(obs)
            self._last_action = None
            self._initial_update(self.history)
        else:
            self.history = self.history.extend(action, obs)
            self._update(self.history)

        update_time = time.time() - start_time
        self._statistics["update_time"] = update_time
        self._log_info1(f"Update time = {update_time:.4f}s")

    def _initial_update(self, history: H.AgentHistory):
        h_node = self.traverse(history)
        _, init_obs = history.get_last_step()
        s_b_0 = self.model.get_agent_initial_belief(self.ego_agent, init_obs)

        hps_b_0 = B.ParticleBelief()
        for _ in range(self.num_sims + self._extra_particles):
            state = s_b_0.sample()
            joint_obs = self.model.sample_initial_obs(state)
            joint_history = H.JointHistory.get_init_history(
                self.num_agents, joint_obs
            )
            policy_state = self._sample_other_policy_prior()
            hp_state = HPS.HistoryPolicyState(
                state, joint_history, policy_state
            )
            hps_b_0.add_particle(hp_state)

        h_node.belief = hps_b_0

    def _update(self, history: H.AgentHistory) -> None:
        self._prune(history)
        self._reinvigorate(history)

    def _prune(self, history: H.AgentHistory):
        self._log_debug("Pruning histories")
        # prune last step in tree
        # remove all actions and obs from parent nodes that are not in
        # last step of history
        h_node = self.traverse(history)
        last_action, _ = history.get_last_step()
        parent_obs_node = h_node.parent.parent
        parent_action_node = parent_obs_node.get_child(last_action)
        parent_obs_node.children = [parent_action_node]
        parent_action_node.children = [h_node]

    #######################################################
    # SEARCH
    #######################################################

    def get_action(self) -> M.Action:
        self._log_info1(f"Searching for num_sims={self.num_sims}")
        start_time = time.time()

        root = self.traverse(self.history)
        if len(root.children) == 0:
            self.expand(root, self.history)

        root_b = root.belief
        for _ in range(self.num_sims):
            hp_state: HPS.HistoryPolicyState = root_b.sample()
            self._rollout_policy.reset_history(self.history)
            self.simulate(hp_state, root, 0)
            root.n += 1

        search_time = time.time() - start_time
        search_time_per_sim = search_time / self.num_sims
        self._statistics["search_time"] += search_time
        self._log_info1(f"{search_time=:.2f} s, {search_time_per_sim=:.5f}")

        return self.get_action_by_history(self.history)

    def simulate(self,
                 hp_state: HPS.HistoryPolicyState,
                 obs_node: Node,
                 depth: int) -> float:
        """Run Monte-Carlo Simulation in tree """
        if self._search_depth_limit_reached(depth):
            return 0.0

        if len(obs_node.children) == 0:
            agent_history = hp_state.history.get_agent_history(self.ego_agent)
            self.expand(obs_node, agent_history)
            return self.rollout(hp_state, depth)

        joint_action = self._get_joint_sim_action(hp_state, obs_node)

        joint_step = self.model.step(hp_state.state, joint_action)
        joint_obs = joint_step.observations

        ego_action = joint_action[self.ego_agent]
        ego_obs = joint_obs[self.ego_agent]
        ego_reward = joint_step.rewards[self.ego_agent]

        new_history = hp_state.history.extend(joint_action, joint_obs)
        next_pi_state = hp_state.other_policies
        next_hp_state = HPS.HistoryPolicyState(
            joint_step.next_state, new_history, next_pi_state
        )

        action_node = obs_node.get_child(ego_action)
        child_obs_node = action_node.get_child(ego_obs)

        child_obs_node.n += 1
        child_obs_node.belief.add_particle(next_hp_state)

        if not joint_step.done:
            self._update_policies(joint_action, joint_obs, next_pi_state)
            ego_reward += self.gamma * self.simulate(
                next_hp_state, child_obs_node, depth+1
            )

        action_node.n += 1
        action_node.v += (ego_reward - action_node.v) / action_node.n
        return ego_reward

    def rollout(self, hp_state: HPS.HistoryPolicyState, depth: int) -> float:
        """Run Monte-Carlo Rollout """
        if self._search_depth_limit_reached(depth):
            return self._rollout_policy.get_value(None)

        joint_action = self._get_joint_rollout_action(hp_state)
        joint_step = self.model.step(hp_state.state, joint_action)
        joint_obs = joint_step.observations

        reward = joint_step.rewards[self.ego_agent]
        if joint_step.done:
            return reward

        new_history = hp_state.history.extend(joint_action, joint_obs)
        next_hp_state = HPS.HistoryPolicyState(
            joint_step.next_state, new_history, hp_state.other_policies
        )

        self._update_policies(joint_action, joint_obs, hp_state.other_policies)

        return reward + self.gamma * self.rollout(next_hp_state, depth+1)

    def _search_depth_limit_reached(self, depth: int) -> bool:
        return (
            depth > self._depth_limit
            or (
                self._step_limit is not None
                and depth + self._step_num > self._step_limit
            )
        )

    def _reset_policies(self,
                        joint_history: H.JointHistory,
                        policy_state: HPS.PolicyState):
        other_policies = self._get_other_policies(policy_state)
        for i in range(self.num_agents):
            h_i = joint_history.get_agent_history(i)
            if i == self.ego_agent:
                pi = self._rollout_policy
            else:
                pi = other_policies[i]
            pi.reset_history(h_i)

    def _update_policies(self,
                         joint_action: M.JointAction,
                         joint_obs: M.JointObservation,
                         policy_state: HPS.PolicyState):
        other_policies = self._get_other_policies(policy_state)
        for i in range(self.num_agents):
            a_i = joint_action[i]
            o_i = joint_obs[i]
            if i == self.ego_agent:
                pi = self._rollout_policy
            else:
                pi = other_policies[i]
            pi.update(a_i, o_i)

    #######################################################
    # ACTION SELECTION
    #######################################################

    def get_action_by_history(self, history: H.AgentHistory) -> M.Action:
        obs_node = self.traverse(history)
        return self.get_action_from_node(obs_node, history, True)

    def get_action_from_node(self,
                             obs_node: Node,
                             history: H.AgentHistory,
                             greedy: bool = False) -> M.Action:
        """Get action from given node in policy tree

        If greedy then selects action with highest value
        else uses UCT action selection.
        """
        if len(obs_node.children) < len(self.action_space):
            self.expand(obs_node, history)

        if obs_node.n == 0:
            return random.choice(self.action_space)

        log_n = math.log(obs_node.n)
        max_v = -float('inf')
        max_action = obs_node.children[0].h
        for action_node in obs_node.children:
            if action_node.n == 0:
                if greedy:
                    continue
                return action_node.h
            action_v = action_node.v
            if not greedy:
                # add exploration bonus based on relative visit count
                action_v += self._uct_c * math.sqrt(log_n / action_node.n)
            if action_v > max_v:
                max_v = action_v
                max_action = action_node.h
        return max_action

    def _get_joint_sim_action(self,
                              hp_state: HPS.HistoryPolicyState,
                              obs_node: Node) -> M.JointAction:
        # Assumes state of other agent policies are primed
        h_i = hp_state.history.get_agent_history(self.ego_agent)
        ego_action = self.get_action_from_node(obs_node, h_i, False)
        return self._get_joint_action(hp_state, ego_action)

    def _get_joint_rollout_action(self,
                                  hp_state: HPS.HistoryPolicyState
                                  ) -> M.JointAction:
        # Assumes state of rollout policy and other agent policies are primed
        ego_action = self._rollout_policy.get_action()
        return self._get_joint_action(hp_state, ego_action)

    def _get_joint_action(self,
                          hp_state: HPS.HistoryPolicyState,
                          ego_action: M.Action) -> M.JointAction:
        agent_actions = []
        other_policies = self._get_other_policies(hp_state.other_policies)
        for i in range(self.num_agents):
            if i == self.ego_agent:
                a_i = ego_action
            else:
                pi = other_policies[i]
                a_i = pi.get_action()
            agent_actions.append(a_i)
        return tuple(agent_actions)

    def get_pi(self,
               history: Optional[H.AgentHistory] = None
               ) -> parts.ActionDist:
        """Get agent's distribution over actions for a given history.

        Returns the softmax distribution over actions with temperature=1.0
        (see POSGMCP.sample_action() function for details). This is
        used as it incorporates uncertainty based on visit counts for a given
        history.
        """
        if history is None:
            history = self.history

        obs_node = self.traverse(history)

        if obs_node.n == 0 or len(obs_node.children) == 0:
            # uniform policy
            num_actions = len(self.action_space)
            pi = {a: 1.0 / num_actions for a in self.action_space}
            return pi

        obs_n_sqrt = math.sqrt(obs_node.n)
        temp = 1.0
        pi = {
            a_node.h: math.exp(a_node.n**temp / obs_n_sqrt)
            for a_node in obs_node.children
        }

        a_probs_sum = sum(pi.values())
        for a in self.action_space:
            if a not in pi:
                pi[a] = 0.0
            pi[a] /= a_probs_sum

        return pi

    #######################################################
    # GENERAL METHODS
    #######################################################

    def traverse(self, history: H.AgentHistory) -> Node:
        """Traverse policy tree and return node corresponding to history """
        h_node = self.root
        for (a, o) in history:
            h_node = h_node.get_child(a).get_child(o)
        return h_node

    def expand(self, obs_node: Node, hist: H.AgentHistory):
        """Add action children to observation node in tree """
        action_init_vals = self._rollout_policy.get_action_init_values(hist)
        for action in self.action_space:
            if obs_node.has_child(action):
                continue
            v_init, n_init = action_init_vals[action]
            action_node = Node(
                action, obs_node, B.ParticleBelief(), v_init, n_init
            )
            obs_node.children.append(action_node)

    #######################################################
    # BELIEF REINVIGORATION
    #######################################################

    def _reinvigorate(self,
                      history: H.AgentHistory,
                      h_node: Optional[Node] = None,
                      target_node_size: Optional[int] = None) -> None:
        # This function wraps the _reinvigorate_belief function and times it
        # This is necessary since the _reinvigorate_belief function can call
        # itself recursively
        self._log_debug1("Reinvigorate")
        start_time = time.time()

        if h_node is None:
            h_node = self.traverse(history)

        if target_node_size is None:
            target_node_size = self.num_sims + self._extra_particles

        self._reinvigorate_belief(history, target_node_size, h_node)

        reinvig_time = time.time() - start_time
        self._statistics["reinvigoration_time"] += reinvig_time
        self._log_debug1(f"Reinvigorate: time={reinvig_time:.6f}")

    def _reinvigorate_belief(self,
                             history: H.AgentHistory,
                             target_node_size: int,
                             h_node: Node):
        """The main belief reinvigoration function.

        The general reinvigoration process:
        1. check belief needs to be reinvigorated (e.g. it's not a root belief)
        2. Reinvigorate node using rejection sampling/custom function for fixed
           number of tries
        3. if desired number of particles not sampled using rejection sampling/
           custom function then sample remaining particles using sampling
           without rejection
        """
        h_node_size = h_node.belief.size()
        if h_node_size is None:
            return

        num_particles = target_node_size - h_node_size
        if num_particles <= 0:
            return

        parent_obs_node = h_node.parent.parent    # type: ignore
        assert parent_obs_node is not None

        action, obs = history[-1]
        self._reinvigorator.reinvigorate(
            self.ego_agent,
            h_node.belief,   # type: ignore
            action,
            obs,
            num_particles=num_particles,
            parent_belief=parent_obs_node.belief,
            joint_action_fn=self._get_joint_action,
            **{
                "use_rejected_samples": True   # used for rejection sampling
            }
        )

    #######################################################
    # Logging
    #######################################################

    def _format_msg(self, msg: str):
        return f"i={self.ego_agent} {msg}"

    def __str__(self):
        return f"{self.__class__.__name__}"

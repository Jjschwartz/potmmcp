from typing import Optional, List, Any, Tuple, Dict

from ray import rllib

import posggym.model as M

import posgmcp.history as H
import posgmcp.policy as policy_lib

from baposgmcp import parts


HiddenState = List[Any]


class RllibPolicy(policy_lib.BasePolicy):
    """A Rllib Policy

    This class essentially acts as an interface between BA-POSGMCP and an
    Rlib Policy class (ray.rllib.policy.policy.Policy)
    """

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 policy: rllib.policy.policy.Policy,
                 **kwargs):
        super().__init__(model, ego_agent, gamma, **kwargs)

        self._policy = policy

        self._last_action: Optional[M.Action] = None
        self._last_obs: Optional[M.Observation] = None
        self._last_hidden_state = self._get_initial_hidden_state()
        self._last_pi_info: Dict[str, Any] = {}

    def step(self, obs: M.Observation) -> M.Action:
        self.update(self._last_action, obs)
        self._last_action = self.get_action()
        return self._last_action

    def get_action(self) -> M.Action:
        return self._last_action

    def get_action_by_history(self, history: H.AgentHistory) -> M.Action:
        _, _, a_t, _ = self._unroll_history(history)
        return a_t

    def get_pi(self,
               history: Optional[H.AgentHistory] = None
               ) -> parts.ActionDist:
        # TODO
        return {}

    def update(self, action: Optional[M.Action], obs: M.Observation) -> None:
        self._last_obs = obs
        output = self._compute_action(
            obs, self._last_hidden_state, self._last_action, explore=False
        )
        self._last_action = output[0]
        self._last_hidden_state = output[1]
        self._last_pi_info = output[2]

    def reset(self) -> None:
        super().reset()
        self._last_obs = None
        self._last_hidden_state = self._get_initial_hidden_state()
        self._last_pi_info = {}

    def reset_history(self, history: H.AgentHistory) -> None:
        super().reset_history(history)
        output = self._unroll_history(history)
        self._last_obs = output[0]
        self._last_hidden_state = output[1]
        self._last_action = output[2]
        self._last_pi_info = output[3]

    def _get_initial_hidden_state(self) -> HiddenState:
        return self._policy.get_initial_state()

    def _compute_action(self,
                        obs: M.Observation,
                        h_tm1: HiddenState,
                        last_action: M.Action,
                        explore: bool = False
                        ) -> Tuple[M.Action, HiddenState, Dict[str, Any]]:
        output = self._policy.compute_single_action(
            obs, h_tm1, prev_action=last_action, explore=explore
        )
        return output

    def _unroll_history(self,
                        history: H.AgentHistory
                        ) -> Tuple[
                            M.Observation,
                            HiddenState,
                            M.Action,
                            Dict[str, Any]
                        ]:
        h_tm1 = self._get_initial_hidden_state()
        info_tm1: Dict[str, Any] = {}
        a_tp1, o_t = history[-1]

        for (a_t, o_t) in history:
            a_tp1, h_tm1, info_tm1 = self._compute_action(o_t, h_tm1, a_t)

        h_t, info_t = h_tm1, info_tm1
        # returns:
        # o_t - the final observation in the history
        # h_t - the hidden state after processing o_t, a_t, h_tm1
        # a_tp1 - the next action to perform after processing o_t, a_t, h_tm1
        # info_t - the info returned after processing o_t, a_t, h_tm1
        return o_t, h_t, a_tp1, info_t

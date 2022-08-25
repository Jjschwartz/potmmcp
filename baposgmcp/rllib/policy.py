import abc
import time
from typing import Optional, List, Any, Tuple, Dict

from ray import rllib

import posggym.model as M

import baposgmcp.policy as P
from baposgmcp.history import AgentHistory

from baposgmcp.rllib import utils


RllibHiddenState = List[Any]


_ACTION_DIST_INPUTS = "action_dist_inputs"
_ACTION_PROB = "action_prob"
_ACTION_LOGP = "action_logp"


class RllibPolicy(P.BasePolicy):
    """A Rllib Policy.

    This class essentially acts as an interface between BA-POSGMCP and an
    Rlib Policy class (ray.rllib.policy.policy.Policy)
    """

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 policy: rllib.policy.policy.Policy,
                 policy_id: Optional[str] = None,
                 preprocessor: Optional[utils.ObsPreprocessor] = None,
                 **kwargs):
        super().__init__(
            model,
            ego_agent,
            gamma,
            policy_id=policy_id,
            **kwargs
        )

        self._policy = policy
        if preprocessor is None:
            preprocessor = utils.identity_preprocessor
        self._preprocessor = preprocessor

        self._last_action: Optional[M.Action] = None
        self._last_obs: Optional[M.Observation] = None
        self._last_hidden_state = self._get_initial_hidden_state()
        self._last_pi_info: Dict[str, Any] = {}

    def step(self, obs: M.Observation) -> M.Action:
        self._log_info1(f"Step obs={obs}")
        start_time = time.time()
        self.update(self._last_action, obs)
        self._last_action = self.get_action()
        self._log_info1(f"Step time = {time.time() - start_time:.4f}s")
        return self._last_action

    def get_action(self) -> M.Action:
        return self._last_action

    def get_action_by_history(self, history: AgentHistory) -> M.Action:
        _, _, a_t, _ = self._unroll_history(history)
        return a_t

    def get_action_by_hidden_state(self,
                                   hidden_state: P.PolicyHiddenState
                                   ) -> M.Action:
        return hidden_state["last_action"]

    def get_pi(self,
               history: Optional[AgentHistory] = None
               ) -> P.ActionDist:
        return self._get_pi_from_info(self._get_info(history))

    def get_pi_from_hidden_state(self,
                                 hidden_state: P.PolicyHiddenState
                                 ) -> P.ActionDist:
        return self._get_pi_from_info(hidden_state["last_pi_info"])

    @abc.abstractmethod
    def _get_pi_from_info(self, info: Dict[str, Any]) -> P.ActionDist:
        """Get policy distribution from info dict."""

    def get_value(self, history: Optional[AgentHistory]) -> float:
        return self._get_value_from_info(self._get_info(history))

    def get_value_by_hidden_state(self,
                                  hidden_state: P.PolicyHiddenState) -> float:
        return self._get_value_from_info(hidden_state["last_pi_info"])

    @abc.abstractmethod
    def _get_value_from_info(self, info: Dict[str, Any]) -> float:
        """Get value from info dict."""

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
        self._log_info1("Reset")
        self._last_obs = None
        self._last_hidden_state = self._get_initial_hidden_state()
        self._last_pi_info = {}

    def reset_history(self, history: AgentHistory) -> None:
        super().reset_history(history)
        output = self._unroll_history(history)
        self._last_obs = output[0]
        self._last_hidden_state = output[1]
        self._last_action = output[2]
        self._last_pi_info = output[3]

    def get_next_hidden_state(self,
                              hidden_state: P.PolicyHiddenState,
                              action: M.Action,
                              obs: M.Observation,
                              explore: Optional[bool] = None
                              ) -> P.PolicyHiddenState:
        next_hidden_state = super().get_next_hidden_state(
            hidden_state, action, obs
        )
        h_tm1 = hidden_state["last_hidden_state"]
        output = self._compute_action(obs, h_tm1, action, explore=explore)
        next_hidden_state["last_obs"] = obs
        next_hidden_state["last_action"] = output[0]
        next_hidden_state["last_hidden_state"] = output[1]
        next_hidden_state["last_pi_info"] = output[2]
        return next_hidden_state

    def get_initial_hidden_state(self) -> P.PolicyHiddenState:
        hidden_state = super().get_initial_hidden_state()
        hidden_state["last_obs"] = None
        hidden_state["last_hidden_state"] = self._get_initial_hidden_state()
        hidden_state["last_action"] = None
        hidden_state["last_pi_info"] = {}
        return hidden_state

    def get_hidden_state(self) -> P.PolicyHiddenState:
        hidden_state = super().get_hidden_state()
        hidden_state["last_obs"] = self._last_obs
        hidden_state["last_hidden_state"] = self._last_hidden_state
        hidden_state["last_pi_info"] = self._last_pi_info
        return hidden_state

    def set_hidden_state(self, hidden_state: P.PolicyHiddenState):
        super().set_hidden_state(hidden_state)
        self._last_obs = hidden_state["last_obs"]
        self._last_hidden_state = hidden_state["last_hidden_state"]
        self._last_pi_info = hidden_state["last_pi_info"]

    def _get_initial_hidden_state(self) -> RllibHiddenState:
        return self._policy.get_initial_state()

    def _compute_action(self,
                        obs: M.Observation,
                        h_tm1: RllibHiddenState,
                        last_action: M.Action,
                        explore: Optional[bool] = None
                        ) -> Tuple[M.Action, RllibHiddenState, Dict[str, Any]]:
        obs = self._preprocessor(obs)
        output = self._policy.compute_single_action(
            obs, h_tm1, prev_action=last_action, explore=explore
        )
        return output

    def _compute_actions(self,
                         obs_batch: List[M.Observation],
                         h_tm1_batch: List[RllibHiddenState],
                         last_action_batch: List[M.Action],
                         explore: bool = False
                         ) -> List[
                             Tuple[M.Action, RllibHiddenState, Dict[str, Any]]
                         ]:
        obs_batch = [self._preprocessor[o] for o in obs_batch]
        output = self._policy.compute_actions(
            obs_batch,
            h_tm1_batch,
            prev_action_batch=last_action_batch,
            explore=explore
        )
        actions = [x[0] for x in output]
        h_t_batch = [x[1] for x in output]
        info_batch = [x[2] for x in output]
        return (actions, h_t_batch, info_batch)

    def _unroll_history(self,
                        history: AgentHistory
                        ) -> Tuple[
                            M.Observation,
                            RllibHiddenState,
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

    def _get_info(self, history: Optional[AgentHistory]) -> Dict[str, Any]:
        if history is None:
            return self._last_pi_info
        _, _, _, info = self._unroll_history(history)
        return info


class PPORllibPolicy(RllibPolicy):
    """A PPO Rllib Policy."""

    VF_PRED = "vf_preds"

    def _get_pi_from_info(self, info: Dict[str, Any]) -> P.ActionDist:
        probs = utils.numpy_softmax(info[_ACTION_DIST_INPUTS])
        return {a: probs[a] for a in range(len(probs))}

    def _get_value_from_info(self, info: Dict[str, Any]) -> float:
        return info[self.VF_PRED]

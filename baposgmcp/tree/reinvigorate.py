"""Functions and classes for belief reinvigoration."""
import abc
from typing import Callable, List

import posggym.model as M

import baposgmcp.hps as H
import baposgmcp.tree.belief as B


# TODO Update this to use Historypolicystate rather than HistoryState


class BeliefReinvigorator(abc.ABC):
    """Abstract class for implementing a belief reinvigorator.."""

    @abc.abstractmethod
    def reinvigorate(self,
                     agent_id: M.AgentID,
                     belief: B.ParticleBelief,
                     action: M.Action,
                     obs: M.Observation,
                     num_particles: int,
                     parent_belief: M.Belief,
                     joint_action_fn: Callable,
                     **kwargs):
        """Reinvigorate belief given action performed and observation recieved.

        In general this involves adding additional particles to the belief that
        are consistent with the action and observation.

        Arguments:
        ---------
        agent_id : M.AgentID
            ID of the agent to reinvigorate belief of
        belief : B.ParticleBelief
            The belief to reinvigorate given last action and observation
        action : M.Action
            Action performed by agent
        obs : M.Observation
            The observation recieved by agent
        num_particles : int
            the number of additional particles to sample
        parent_belief : M.Belief
            the parent belief of the belief being reinvigorated
        joint_action_fn : Callable[[H.HistoryState, M.Action], M.JointAction]
            joint action selection function

        """


class BeliefRejectionSampler(BeliefReinvigorator):
    """Reinvigorates a belief using rejection sampling.

    Reinvigorate function takes additional optional and requires kwargs:

    'use_rejected_samples': bool, Optional
        whether to use sampling without rejection to ensure 'num_particles'
        additional particles are added to the belief in the case that
        'num_particles' valid particles aren't sampled within 'sample_limit'
        samples using rejection sampling (default=False)
    """

    def __init__(self,
                 model: M.POSGModel,
                 sample_limit: int = 1000):
        self._model = model
        self._sample_limit = sample_limit

    def reinvigorate(self,
                     agent_id: M.AgentID,
                     belief: B.ParticleBelief,
                     action: M.Action,
                     obs: M.Observation,
                     num_particles: int,
                     parent_belief: M.Belief,
                     joint_action_fn: Callable,
                     **kwargs):
        new_particles = self._rejection_sample(
            agent_id,
            action,
            obs,
            parent_belief=parent_belief,
            num_samples=num_particles,
            joint_action_fn=joint_action_fn,
            use_rejected_samples=kwargs.get("use_rejected_samples", False)
        )
        belief.add_particles(new_particles)

    def _rejection_sample(self,
                          agent_id: M.AgentID,
                          action: M.Action,
                          obs: M.Observation,
                          parent_belief: B.BaseParticleBelief,
                          num_samples: int,
                          joint_action_fn: Callable,
                          use_rejected_samples: bool) -> List[H.HistoryState]:
        sample_count = 0
        retry_count = 0
        rejected_samples = []
        samples = []

        while (
            sample_count < num_samples
            and retry_count < max(num_samples, self._sample_limit)
        ):
            h_state = parent_belief.sample()
            joint_action = joint_action_fn(h_state, action)
            joint_step = self._model.step(h_state.state, joint_action)

            joint_obs = joint_step.observations
            new_history = h_state.history.extend(joint_action, joint_obs)
            next_h_state = H.HistoryState(joint_step.state, new_history)

            if joint_obs[agent_id] == obs:
                samples.append(next_h_state)
                sample_count += 1
            else:
                if use_rejected_samples:
                    rejected_samples.append(next_h_state)
                retry_count += 1

        if sample_count < num_samples and use_rejected_samples:
            num_missing = num_samples - sample_count
            samples.extend(rejected_samples[:num_missing])

        return samples


class BeliefRandomSampler(BeliefReinvigorator):
    """Reinvigorates a belief using random sampling.

    This is essentially rejection sampling without checking for observation
    consistency, instead it simply replaces the agents sampled observation (in
    the sampled joint observation) with the actual agent observation.
    """

    def __init__(self, model: M.POSGModel):
        self._model = model

    def reinvigorate(self,
                     agent_id: M.AgentID,
                     belief: B.ParticleBelief,
                     action: M.Action,
                     obs: M.Observation,
                     num_particles: int,
                     parent_belief: M.Belief,
                     joint_action_fn: Callable,
                     **kwargs):
        new_particles = self._random_sample(
            agent_id,
            action,
            obs,
            parent_belief=parent_belief,
            num_samples=num_particles,
            joint_action_fn=joint_action_fn,
        )
        belief.add_particles(new_particles)

    def _random_sample(self,
                       agent_id: M.AgentID,
                       action: M.Action,
                       obs: M.Observation,
                       parent_belief: B.BaseParticleBelief,
                       num_samples: int,
                       joint_action_fn: Callable
                       ) -> List[H.HistoryState]:
        samples = []
        for _ in range(num_samples):
            h_state = parent_belief.sample()
            joint_action = joint_action_fn(h_state, action)
            joint_step = self._model.step(h_state.state, joint_action)

            # replace sampled agent obs with real obs
            joint_obs = joint_step.observations
            tmp = list(joint_obs)
            tmp[agent_id] = obs
            joint_obs = tuple(tmp)

            new_history = h_state.history.extend(joint_action, joint_obs)
            next_h_state = H.HistoryState(joint_step.state, new_history)
            samples.append(next_h_state)

        return samples

"""Functions and classes for belief reinvigoration in BAPOSGMCP """
from typing import Callable, List

import posggym.model as M

import posgmcp.belief as B
import posgmcp.history as H
from posgmcp.reinvigorate import BeliefReinvigorator

import baposgmcp.hps as HPS


class BABeliefRejectionSampler(BeliefReinvigorator):
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
            hp_state = parent_belief.sample()
            joint_action = joint_action_fn(hp_state, action)
            joint_step = self._model.step(hp_state.state, joint_action)
            joint_obs = joint_step.observations

            new_history = hp_state.history.extend(joint_action, joint_obs)
            next_hp_state = HPS.HistoryPolicyState(
                joint_step.state, new_history, hp_state.other_policies
            )

            if joint_obs[agent_id] == obs:
                samples.append(next_hp_state)
                sample_count += 1
            else:
                if use_rejected_samples:
                    rejected_samples.append(next_hp_state)
                retry_count += 1

        if sample_count < num_samples and use_rejected_samples:
            num_missing = num_samples - sample_count
            samples.extend(rejected_samples[:num_missing])

        return samples


class BABeliefRandomSampler(BeliefReinvigorator):
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
            hp_state = parent_belief.sample()
            joint_action = joint_action_fn(hp_state, action)
            joint_step = self._model.step(hp_state.state, joint_action)
            joint_obs = joint_step.observations

            # replace sampled agent obs with real obs
            tmp = list(joint_obs)
            tmp[agent_id] = obs
            joint_obs = tuple(tmp)

            new_history = hp_state.history.extend(joint_action, joint_obs)
            next_hp_state = HPS.HistoryPolicyState(
                joint_step.state, new_history, hp_state.other_policies
            )
            samples.append(next_hp_state)

        return samples

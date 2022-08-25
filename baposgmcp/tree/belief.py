"""Belief data structures."""
import abc
import random
from typing import Optional, Dict, Sequence, List

import posggym.model as M

from baposgmcp.policy import PolicyID
from baposgmcp.tree.hps import HistoryPolicyState


class BaseParticleBelief(M.Belief, abc.ABC):
    """An abstract particle belief.

    Represents a belief as a set of particles
    """

    @abc.abstractmethod
    def is_depleted(self) -> bool:
        """Return true if belief is depleted, so cannot be sampled."""

    @abc.abstractmethod
    def add_particle(self, state: M.State) -> None:
        """Add a single state particle to the belief."""

    @abc.abstractmethod
    def add_particles(self, states: Sequence[M.State]) -> None:
        """Add a multiple state particle to the belief."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Delete any stored particles."""

    @abc.abstractmethod
    def size(self) -> Optional[int]:
        """Get the number of particles stored in belief.

        Returns None if belief has infinite size (e.g. uses a function to
        generate particles)
        """


class ParticleBelief(BaseParticleBelief):
    """A belief represented by state particles."""

    def __init__(self):
        super().__init__()
        self.particles = []

    def sample(self) -> M.State:
        return random.choice(self.particles)

    def sample_k(self, k: int) -> Sequence[M.State]:
        return random.choices(self.particles, k=k)

    def add_particle(self, state: M.State):
        self.particles.append(state)

    def add_particles(self, states: Sequence[M.State]):
        self.particles.extend(states)

    def is_depleted(self) -> bool:
        return len(self.particles) == 0

    def size(self) -> Optional[int]:
        return len(self.particles)

    def clear(self) -> None:
        self.particles.clear()

    def get_dist(self) -> Dict[M.State, float]:
        unique_particles = list(set(self.particles))
        dist = {}
        prob_sum = 0.0
        for particle in unique_particles:
            count = self.particles.count(particle)
            prob = count / self.size()
            dist[particle] = prob
            prob_sum += prob

        if prob_sum < 1.0:
            for state, prob in dist.items():
                dist[state] = prob / prob_sum
        return dist


class HPSParticleBelief(BaseParticleBelief):
    """A belief over History-Policy-States represented by particles."""

    def __init__(self, agent_policy_id_map: Dict[M.AgentID, List[PolicyID]]):
        super().__init__()
        self._particles: List[HistoryPolicyState] = []
        self._agent_policy_particles: Dict[
            M.AgentID, Dict[PolicyID, List[HistoryPolicyState]]
        ] = {}

        for i, policy_ids in agent_policy_id_map.items():
            self._agent_policy_particles[i] = {
                pi_id: [] for pi_id in policy_ids
            }

    def sample(self) -> HistoryPolicyState:
        return random.choice(self._particles)

    def sample_k(self, k: int) -> Sequence[HistoryPolicyState]:
        return random.choices(self._particles, k=k)

    def sample_agent_policy(self,
                            agent_id: M.AgentID,
                            policy_id: PolicyID) -> HistoryPolicyState:
        """Sample particle that contains given policy id for given agent.

        Will raise IndexError if no particles associated with given
        agent_id, policy id pair.

        Will raise KeyError if agent id or policy id are not part of this
        belief.
        """
        return random.choice(self._agent_policy_particles[agent_id][policy_id])

    def sample_k_agent_policy(self,
                              k: int,
                              agent_id: M.AgentID,
                              policy_id: PolicyID
                              ) -> Sequence[HistoryPolicyState]:
        """Sample k particles that contains given policy id for given agent.

        Will raise IndexError if no particles associated with given
        agent_id, policy id pair.

        Will raise KeyError if agent id or policy id are not part of this
        belief.
        """
        return random.choices(
            self._agent_policy_particles[agent_id][policy_id], k=k
        )

    def sample_policy_ids(self) -> Dict[M.AgentID, PolicyID]:
        """Sample policy ID for each agent from belief.

        Samples policy IDs based on the proportion of particles associated with
        each policy ID for a given agent.
        """
        policy_ids = {}
        for i, pi_particle_map in self._agent_policy_particles.items():
            pi_weights = [
                len(pi_particles) for pi_particles in pi_particle_map.values()
            ]
            policy_ids[i] = random.choices(
                list(pi_particle_map), weights=pi_weights, k=1
            )[0]
        return policy_ids

    def sample_k_correlated(self, k: int) -> Sequence[HistoryPolicyState]:
        """Sample k correlated particles.

        Correlated particles have the same policy ID for across all particles
        for each agent. Note, policies may be different between agents but will
        be the same across all particles for a given agent.

        This is useful for batched inference when running simulations in
        parallel.
        """
        # TODO Currently this only works for two agent settings
        # to support more than two agents will require correlated sampling
        assert len(self._agent_policy_particles) == 1

        policy_ids = self.sample_policy_ids()
        for i, pi_id in policy_ids.items():
            return self.sample_k_agent_policy(k, i, pi_id)

    def add_particle(self, state: HistoryPolicyState):
        self._particles.append(state)
        pi_state = state.other_policies
        for i, pi_id in enumerate(pi_state):
            if i not in self._agent_policy_particles:
                continue
            self._agent_policy_particles[i][pi_id].append(state)

    def add_particles(self, states: Sequence[HistoryPolicyState]):
        for s in states:
            self.add_particle(s)

    def is_depleted(self) -> bool:
        return len(self._particles) == 0

    def size(self) -> Optional[int]:
        return len(self._particles)

    def clear(self) -> None:
        self._particles.clear()
        for i, pi_map in self._agent_policy_particles.items():
            for particles in pi_map.values():
                particles.clear()

    def get_dist(self) -> Dict[HistoryPolicyState, float]:
        unique_particles = list(set(self._particles))
        dist = {}
        prob_sum = 0.0
        for particle in unique_particles:
            count = self._particles.count(particle)
            prob = count / self.size()
            dist[particle] = prob
            prob_sum += prob

        if prob_sum < 1.0:
            for state, prob in dist.items():
                dist[state] = prob / prob_sum
        return dist

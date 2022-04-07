"""Belief data structures."""
import abc
import random
from typing import Optional, Dict, Sequence, Callable

import posggym.model as M


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


class InitialParticleBelief(BaseParticleBelief):
    """The initial particle belief for a problem.

    Uses a function for generating particles.
    """

    def __init__(self,
                 initial_belief_fn: Callable[[], M.State],
                 dist_res: int = 100):
        self._i_fn = initial_belief_fn
        self._dist_res = dist_res

    def sample(self) -> M.State:
        return self._i_fn()

    def sample_k(self, k: int) -> Sequence[M.State]:
        samples = []
        for _ in range(k):
            samples.append(self.sample())
        return samples

    def is_depleted(self) -> bool:
        return False

    def add_particle(self, state: M.State) -> None:
        return

    def add_particles(self, states: Sequence[M.State]) -> None:
        return

    def clear(self) -> None:
        return

    def size(self) -> Optional[int]:
        return None

    def get_dist(self) -> Dict[M.State, float]:
        return self.sample_belief_dist(self._dist_res)


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

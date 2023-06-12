import abc
import time
from collections import ChainMap
from typing import Mapping, Any, List, Sequence, Iterable

import numpy as np

import posggym
import posggym.model as M

import posggym_agents.policy as Pi


AgentStatisticsMap = Mapping[M.AgentID, Mapping[str, Any]]


def generate_episode_statistics(trackers: Iterable['Tracker']
                                ) -> AgentStatisticsMap:
    """Generate episode statistics from set of trackers."""
    statistics = combine_statistics([t.get_episode() for t in trackers])
    return statistics


def generate_statistics(trackers: Iterable['Tracker']) -> AgentStatisticsMap:
    """Generate summary statistics from set of trackers."""
    statistics = combine_statistics([t.get() for t in trackers])
    return statistics


def combine_statistics(statistic_maps: Sequence[AgentStatisticsMap]
                       ) -> AgentStatisticsMap:
    """Combine multiple Agent statistic maps into a single one."""
    agent_ids = list(statistic_maps[0].keys())
    return {
        i: dict(ChainMap(*(stat_maps[i] for stat_maps in statistic_maps)))
        for i in agent_ids
    }


def get_default_trackers() -> List['Tracker']:
    """Get the default set of Trackers."""
    return [EpisodeTracker()]


class Tracker(abc.ABC):
    """Generic Tracker Base class."""

    @abc.abstractmethod
    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[Pi.BasePolicy],
             episode_end: bool):
        """Accumulates statistics for a single step."""

    @abc.abstractmethod
    def reset(self):
        """Reset all gathered statistics."""

    @abc.abstractmethod
    def reset_episode(self):
        """Reset all statistics prior to each episode."""

    @abc.abstractmethod
    def get_episode(self) -> AgentStatisticsMap:
        """Aggregate episode statistics for each agent."""

    @abc.abstractmethod
    def get(self) -> AgentStatisticsMap:
        """Aggregate all episode statistics for each agent."""


class EpisodeTracker(Tracker):
    """Tracks episode return and other statistics."""

    def __init__(self):
        # is initialized when step is first called
        self._num_agents = None

        self._num_episodes = 0
        self._current_episode_done = False
        self._current_episode_start_time = time.time()
        # is initialized when step is first called
        self._current_episode_returns = None
        self._current_episode_steps = 0

        self._dones = []
        self._times = []
        self._returns = []
        self._steps = []
        self._outcomes = []

    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[Pi.BasePolicy],
             episode_end: bool):
        if self._num_agents is None:
            self._num_agents = env.n_agents
            self._current_episode_returns = np.zeros(env.n_agents)

        if episode_t == 0:
            return

        _, rewards, done, aux = timestep
        self._current_episode_returns += rewards
        self._current_episode_done = done
        self._current_episode_steps += 1

        if episode_end:
            self._num_episodes += 1
            self._dones.append(self._current_episode_done)
            self._times.append(time.time() - self._current_episode_start_time)
            self._returns.append(self._current_episode_returns)
            self._steps.append(self._current_episode_steps)

            if aux.get("outcome", None) is None:
                outcome = tuple(M.Outcome.NA for _ in range(env.n_agents))
            else:
                outcome = aux["outcome"]
            self._outcomes.append(outcome)

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._dones = []
        self._times = []
        self._returns = []
        self._steps = []
        self._outcomes = []

    def reset_episode(self):
        self._current_episode_done = False
        self._current_episode_start_time = time.time()
        self._current_episode_returns = np.zeros(self._num_agents)
        self._current_episode_steps = 0

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            stats[i] = {
                "episode_number": self._num_episodes,
                "episode_return": self._returns[-1][i],
                "episode_steps": self._steps[-1],
                "episode_outcome": self._outcomes[-1][i],
                "episode_done": self._dones[-1],
                "episode_time": self._times[-1]
            }

        return stats

    def get(self) -> AgentStatisticsMap:
        outcome_counts = {
            k: [0 for _ in range(self._num_agents)] for k in M.Outcome
        }
        for outcome in self._outcomes:
            for i in range(self._num_agents):
                outcome_counts[outcome[i]][i] += 1

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {
                "num_episodes": self._num_episodes,
                "episode_return_mean": np.mean(self._returns, axis=0)[i],
                "episode_return_std": np.std(self._returns, axis=0)[i],
                "episode_return_max": np.max(self._returns, axis=0)[i],
                "episode_return_min": np.min(self._returns, axis=0)[i],
                "episode_steps_mean": np.mean(self._steps),
                "episode_steps_std": np.std(self._steps),
                "episode_time_mean": np.mean(self._times),
                "episode_time_std": np.std(self._times),
                "num_episode_done": np.sum(self._dones)
            }

            for outcome, counts in outcome_counts.items():
                stats[i][f"num_{outcome}"] = counts[i]

        return stats

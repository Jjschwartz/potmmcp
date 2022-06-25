import os
import abc
import time
import csv
import pathlib
from datetime import datetime
from collections import ChainMap
from typing import Mapping, Any, List, Sequence, Iterable, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from prettytable import PrettyTable

import posggym
import posggym.model as M

from baposgmcp import parts
import baposgmcp.tree as tree_lib
import baposgmcp.policy as policy_lib
from baposgmcp.config import BASE_RESULTS_DIR

AgentStatisticsMap = Mapping[M.AgentID, Mapping[str, Any]]
COMPILED_RESULTS_FNAME = "compiled_results.csv"


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


def get_action_dist_distance(dist1: parts.ActionDist,
                             dist2: parts.ActionDist) -> float:
    """Get the Wassersteing distance between two action distributions."""
    # ensure dists are over the same actions or one dist's actions are
    # a subset of the other
    if len(dist1) >= len(dist2):
        actions = list(dist1)
        assert all(a2 in dist1 for a2 in dist2)
    else:
        actions = list(dist2)
        assert all(a1 in dist2 for a1 in dist1)

    probs1 = []
    probs2 = []
    for a in actions:
        probs1.append(dist1.get(a, 0.0))
        probs2.append(dist2.get(a, 0.0))

    return stats.wasserstein_distance(actions, actions, probs1, probs2)


def get_default_trackers(policies: Sequence[policy_lib.BasePolicy]
                         ) -> Sequence['Tracker']:
    """Get the default set of Trackers."""
    num_agents = len(policies)
    gammas = [pi.gamma for pi in policies]
    trackers = [
        EpisodeTracker(num_agents, gammas),
        SearchTimeTracker(num_agents),
    ]

    return trackers


class Tracker(abc.ABC):
    """Generic Tracker Base class."""

    @abc.abstractmethod
    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy_lib.BasePolicy],
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

    def __init__(self, num_agents: int, discounts: List[float]):
        assert len(discounts) == num_agents
        self._num_agents = num_agents
        self._discounts = np.array(discounts)

        self._num_episodes = 0
        self._current_episode_done = False
        self._current_episode_start_time = time.time()
        self._current_episode_returns = np.zeros(num_agents)
        self._current_episode_discounted_returns = np.zeros(num_agents)
        self._current_episode_steps = 0

        self._dones = []                # type: ignore
        self._times = []                # type: ignore
        self._returns = []              # type: ignore
        self._discounted_returns = []   # type: ignore
        self._steps = []                # type: ignore
        self._outcomes = []             # type: ignore

    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy_lib.BasePolicy],
             episode_end: bool):
        if episode_t == 0:
            return

        _, rewards, done, aux = timestep
        self._current_episode_returns += rewards
        self._current_episode_discounted_returns += (
            self._discounts**self._current_episode_steps * rewards
        )
        self._current_episode_done = done
        self._current_episode_steps += 1

        if episode_end:
            self._num_episodes += 1
            self._dones.append(self._current_episode_done)
            self._times.append(time.time() - self._current_episode_start_time)
            self._returns.append(self._current_episode_returns)
            self._discounted_returns.append(
                self._current_episode_discounted_returns
            )
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
        self._discounted_returns = []
        self._steps = []
        self._outcomes = []

    def reset_episode(self):
        self._current_episode_done = False
        self._current_episode_start_time = time.time()
        self._current_episode_returns = np.zeros(self._num_agents)
        self._current_episode_discounted_returns = np.zeros(self._num_agents)
        self._current_episode_steps = 0

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            stats[i] = {
                "episode_number": self._num_episodes,
                "episode_return": self._returns[-1][i],
                "episode_discounted_return": (
                    self._discounted_returns[-1][i]
                ),
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
                "episode_returns_mean": np.mean(self._returns, axis=0)[i],
                "episode_returns_std": np.std(self._returns, axis=0)[i],
                "episode_returns_max": np.max(self._returns, axis=0)[i],
                "episode_returns_min": np.min(self._returns, axis=0)[i],
                "episode_discounted_returns_mean": (
                    np.mean(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_returns_std": (
                    np.std(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_returns_max": (
                    np.max(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_returns_min": (
                    np.min(self._discounted_returns, axis=0)[i]
                ),
                "episode_steps_mean": np.mean(self._steps),
                "episode_steps_std": np.std(self._steps),
                "episode_times_mean": np.mean(self._times),
                "episode_times_std": np.std(self._times),
                "episode_dones": np.mean(self._dones)
            }

            for outcome, counts in outcome_counts.items():
                stats[i][f"num_outcome_{outcome}"] = counts[i]

        return stats


class SearchTimeTracker(Tracker):
    """Tracks Search, Update, Reinvigoration time in Search Trees."""

    # The list of keys to track from the policies.statistics property
    TIME_KEYS = [
        "search_time",
        "update_time",
        "reinvigoration_time",
        "policy_calls",
        "inference_time"
    ]

    def __init__(self, num_agents: int):
        self._num_agents = num_agents

        self._num_episodes = 0
        self._current_episode_steps = 0
        self._current_episode_times: Dict[
            M.AgentID, Dict[str, List[float]]
        ] = {}

        self._steps: List[int] = []
        self._times: Dict[M.AgentID, Dict[str, List[float]]] = {}

        self.reset()

    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy_lib.BasePolicy],
             episode_end: bool):
        if episode_t == 0:
            return

        self._current_episode_steps += 1

        for i in range(self._num_agents):
            statistics = policies[i].statistics
            for time_key in self.TIME_KEYS:
                self._current_episode_times[i][time_key].append(
                    statistics.get(time_key, 0.0)
                )

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._current_episode_steps)
            for i in range(self._num_agents):
                for k in self.TIME_KEYS:
                    key_step_times = self._current_episode_times[i][k]
                    self._times[i][k].append(np.mean(key_step_times))

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._times = {}
        for i in range(self._num_agents):
            self._times[i] = {k: [] for k in self.TIME_KEYS}

    def reset_episode(self):
        self._current_episode_steps = 0
        self._current_episode_times = {}
        for i in range(self._num_agents):
            self._current_episode_times[i] = {k: [] for k in self.TIME_KEYS}

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            agent_stats = {}
            for key, step_times in self._current_episode_times[i].items():
                agent_stats[key] = np.mean(step_times, axis=0)
            stats[i] = agent_stats

        return stats

    def get(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            agent_stats = {}
            for key, values in self._times[i].items():
                agent_stats[f"{key}_mean"] = np.mean(values, axis=0)
                agent_stats[f"{key}_std"] = np.std(values, axis=0)
            stats[i] = agent_stats
        return stats


class BayesAccuracyTracker(Tracker):
    """Tracks Accuracy between distribution over pis and the true pi.

    Only tracks for BAPOSGMCP policy and if the opponent policy has a policy ID
    that matches a policy ID within the BAPOSGMCP other agent policy
    distribution.

    If track_per_step=True, then outputs accuracy for each episode step in the
    for "bayes_accuracy_<agent_id>_<step_num>". This may generate a lot of
    output, so beware.

    """

    def __init__(self,
                 num_agents: int,
                 track_per_step: bool,
                 step_limit: Optional[int] = None):
        self._num_agents = num_agents
        self._track_per_step = track_per_step
        self._step_limit = step_limit

        self._num_episodes = 0
        self._episode_steps = 0
        self._episode_acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}

        self._steps: List[int] = []
        self._acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}
        self._step_acc: Dict[
            M.AgentID, Dict[M.AgentID, List[List[float]]]
        ] = {}

        self.reset()

    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy_lib.BasePolicy],
             episode_end: bool):
        if episode_t == 0:
            return

        self._episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.BAPOSGMCP):
                continue

            pi_beliefs = tree_lib.get_other_pis_belief(policies[i])
            if pi_beliefs is None:
                continue

            for j in range(self._num_agents):
                if i == j:
                    continue
                policy_id_j = policies[j].policy_id
                acc = pi_beliefs[j].get(policy_id_j, 0.0)
                self._episode_acc[i][j].append(acc)

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._episode_steps)
            for i in range(self._num_agents):
                for j, acc in self._episode_acc[i].items():
                    self._acc[i][j].append(np.mean(acc, axis=0))
                    if self._track_per_step:
                        self._step_acc[i][j].append(acc)

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._acc = {}
        for i in range(self._num_agents):
            self._acc[i] = {j: [] for j in range(self._num_agents) if j != i}
            self._step_acc[i] = {
                j: [] for j in range(self._num_agents) if j != i
            }

    def reset_episode(self):
        self._episode_steps = 0
        self._episode_acc = {}
        for i in range(self._num_agents):
            self._episode_acc[i] = {
                j: [] for j in range(self._num_agents) if j != i
            }

    def get_episode(self) -> AgentStatisticsMap:
        if self._step_limit is not None:
            num_steps = self._step_limit
        else:
            num_steps = 0
            for i in range(self._num_agents):
                for acc in self._episode_acc[i].values():
                    num_steps = max(num_steps, len(acc))

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {f"bayes_accuracy_{i}": np.nan}
            for j, acc in self._episode_acc[i].items():
                stats[i][f"bayes_accuracy_{j}"] = np.mean(acc, axis=0)

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"bayes_accuracy_{i}_{t}"] = np.nan

                for j, acc in self._episode_acc[i].items():
                    for t in range(num_steps):
                        acc_value = np.nan if len(acc) <= t else acc[t]
                        stats[i][f"bayes_accuracy_{j}_{t}"] = acc_value

        return stats

    def get(self) -> AgentStatisticsMap:
        if self._step_limit is not None:
            num_steps = self._step_limit
        else:
            num_steps = 0
            for i in range(self._num_agents):
                for ep_accs in self._step_acc[i].values():
                    max_ep_len = max(len(ep) for ep in ep_accs)
                    num_steps = max(max_ep_len, num_steps)

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {f"bayes_accuracy_{i}": np.nan}
            for j, acc in self._acc[i].items():
                stats[i][f"bayes_accuracy_{j}"] = np.mean(acc, axis=0)

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"bayes_accuracy_{i}_{t}"] = np.nan

                for j, ep_accs in self._step_acc[i].items():
                    accs_by_t = [[] for _ in range(num_steps)]
                    for acc in ep_accs:
                        for t, v in enumerate(acc):
                            accs_by_t[t].append(v)

                    for t in range(num_steps):
                        mean_acc = np.mean(accs_by_t[t])
                        stats[i][f"bayes_accuracy_{j}_{t}"] = mean_acc

        return stats


class BeliefStateAccuracyTracker(Tracker):
    """Tracks accuracy between belief over states and the true state.

    Only tracks for BAPOSGMCP policy

    If track_per_step=True, then outputs accuracy for each episode step in the
    for "belief_accuracy_<agent_id>_<step_num>". This may generate a lot of
    output, so beware.

    """

    def __init__(self,
                 num_agents: int,
                 track_per_step: bool,
                 step_limit: Optional[int] = None):
        self._num_agents = num_agents
        self._track_per_step = track_per_step
        self._step_limit = step_limit

        self._num_episodes = 0
        self._episode_steps = 0
        self._prev_state: M.State = None
        self._episode_acc: Dict[M.AgentID, List[float]] = {}

        self._steps: List[int] = []
        self._acc: Dict[M.AgentID, List[float]] = {}
        self._step_acc: Dict[M.AgentID, List[List[float]]] = {}

        self.reset()

    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy_lib.BasePolicy],
             episode_end: bool):
        if episode_t == 0:
            self._prev_state = env.state
            return

        self._episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.BAPOSGMCP):
                continue

            state_belief = tree_lib.get_state_belief(policies[i])
            if state_belief is None:
                continue

            acc = state_belief.get(self._prev_state, 0.0)
            self._episode_acc[i].append(acc)

        self._prev_state = env.state

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._episode_steps)
            for i in range(self._num_agents):
                self._acc[i].append(np.mean(self._episode_acc[i], axis=0))
                if self._track_per_step:
                    self._step_acc[i].append(self._episode_acc[i])

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._prev_state = None
        self._steps = []
        self._acc = {}
        for i in range(self._num_agents):
            self._acc[i] = []
            self._step_acc[i] = []

    def reset_episode(self):
        self._episode_steps = 0
        self._prev_state = None
        self._episode_acc = {}
        for i in range(self._num_agents):
            self._episode_acc[i] = []

    def get_episode(self) -> AgentStatisticsMap:
        if self._step_limit is not None:
            num_steps = self._step_limit
        else:
            num_steps = 0
            for i in range(self._num_agents):
                num_steps = max(num_steps, len(self._episode_acc[i]))

        stats = {}
        for i in range(self._num_agents):
            acc = self._episode_acc[i]
            stats[i] = {"state_accuracy": np.mean(acc, axis=0)}

            if self._track_per_step:
                for t in range(num_steps):
                    acc_value = np.nan if len(acc) <= t else acc[t]
                    stats[i][f"state_accuracy_{t}"] = acc_value

        return stats

    def get(self) -> AgentStatisticsMap:
        if self._step_limit is not None:
            num_steps = self._step_limit
        else:
            num_steps = 0
            for i in range(self._num_agents):
                max_ep_len = max(len(ep) for ep in self._step_acc[i])
                num_steps = max(max_ep_len, num_steps)

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {"state_accuracy": np.mean(self._acc[i], axis=0)}

            if self._track_per_step:
                ep_accs = self._step_acc[i]
                accs_by_t = [[] for _ in range(num_steps)]
                for acc in ep_accs:
                    for t, v in enumerate(acc):
                        accs_by_t[t].append(v)

                for t in range(num_steps):
                    mean_acc = np.mean(accs_by_t[t])
                    stats[i][f"state_accuracy_{t}"] = mean_acc

        return stats


class BeliefHistoryAccuracyTracker(Tracker):
    """Tracks accuracy of the belief over history of the other agents.

    Only tracks for BAPOSGMCP policy

    If track_per_step=True, then outputs accuracy for each episode step in the
    for "history_accuracy_<agent_id>_<step_num>". This may generate a lot of
    output, so beware.

    """

    def __init__(self,
                 num_agents: int,
                 track_per_step: bool,
                 step_limit: Optional[int] = None):
        self._num_agents = num_agents
        self._track_per_step = track_per_step
        self._step_limit = step_limit

        self._num_episodes = 0
        self._episode_steps = 0
        self._episode_acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}

        self._steps: List[int] = []
        self._acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}
        self._step_acc: Dict[
            M.AgentID, Dict[M.AgentID, List[List[float]]]
        ] = {}

        self.reset()

    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy_lib.BasePolicy],
             episode_end: bool):
        if episode_t == 0:
            return

        self._episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.BAPOSGMCP):
                continue

            h_beliefs = tree_lib.get_other_history_belief(policies[i])
            if h_beliefs is None:
                continue

            for j in range(self._num_agents):
                if i == j:
                    continue
                h_j = policies[j].history
                acc = h_beliefs[j].get(h_j, 0.0)
                self._episode_acc[i][j].append(acc)

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._episode_steps)
            for i in range(self._num_agents):
                for j, acc in self._episode_acc[i].items():
                    self._acc[i][j].append(np.mean(acc, axis=0))
                if self._track_per_step:
                    self._step_acc[i][j].append(acc)

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._acc = {}
        for i in range(self._num_agents):
            self._acc[i] = {j: [] for j in range(self._num_agents) if j != i}
            self._step_acc[i] = {
                j: [] for j in range(self._num_agents) if j != i
            }

    def reset_episode(self):
        self._episode_steps = 0
        self._episode_acc = {}
        for i in range(self._num_agents):
            self._episode_acc[i] = {
                j: [] for j in range(self._num_agents) if j != i
            }

    def get_episode(self) -> AgentStatisticsMap:
        if self._step_limit is not None:
            num_steps = self._step_limit
        else:
            num_steps = 0
            for i in range(self._num_agents):
                for acc in self._episode_acc[i].values():
                    num_steps = max(num_steps, len(acc))

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {f"history_accuracy_{i}": np.nan}
            for j, acc in self._episode_acc[i].items():
                stats[i][f"history_accuracy_{j}"] = np.mean(acc, axis=0)

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"history_accuracy_{i}_{t}"] = np.nan

                for j, acc in self._episode_acc[i].items():
                    for t in range(num_steps):
                        acc_value = np.nan if len(acc) <= t else acc[t]
                        stats[i][f"history_accuracy_{j}_{t}"] = acc_value

        return stats

    def get(self) -> AgentStatisticsMap:
        if self._step_limit is not None:
            num_steps = self._step_limit
        else:
            num_steps = 0
            for i in range(self._num_agents):
                for ep_accs in self._step_acc[i].values():
                    max_ep_len = max(len(ep) for ep in ep_accs)
                    num_steps = max(max_ep_len, num_steps)

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {f"history_accuracy_{i}": np.nan}
            for j, acc in self._acc[i].items():
                stats[i][f"history_accuracy_{j}"] = np.mean(acc, axis=0)

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"history_accuracy_{i}_{t}"] = np.nan

                for j, ep_accs in self._step_acc[i].items():
                    accs_by_t = [[] for _ in range(num_steps)]
                    for acc in ep_accs:
                        for t, v in enumerate(acc):
                            accs_by_t[t].append(v)

                    for t in range(num_steps):
                        mean_acc = np.mean(accs_by_t[t])
                        stats[i][f"history_accuracy_{j}_{t}"] = mean_acc

        return stats


class ActionDistributionDistanceTracker(Tracker):
    """Tracks distance between expected and true policy of the other agents.

    Only tracks for BAPOSGMCP policy.

    This looks specifically at the expected distribution over actions of the
    other agent at the current root node of the BAPOSGMCP policy tree. This is
    an expectation over the history of the other agent and the policy of the
    other agent for the root belief.

    If track_per_step=True, then outputs accuracy for each episode step in the
    for "history_accuracy_<agent_id>_<step_num>". This may generate a lot of
    output, so beware.

    Note this reports the Wassersteing Distance between the expected policy and
    the true agent policy.

    """

    def __init__(self,
                 num_agents: int,
                 track_per_step: bool,
                 step_limit: Optional[int] = None):
        self._num_agents = num_agents
        self._track_per_step = track_per_step
        self._step_limit = step_limit

        self._num_episodes = 0
        self._episode_steps = 0
        self._episode_acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}

        self._steps: List[int] = []
        self._acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}
        self._step_acc: Dict[
            M.AgentID, Dict[M.AgentID, List[List[float]]]
        ] = {}

        self.reset()

    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy_lib.BasePolicy],
             episode_end: bool):
        if episode_t == 0:
            return

        self._episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.BAPOSGMCP):
                continue

            other_action_dists = tree_lib.get_other_agent_action_dist(
                policies[i]
            )
            if other_action_dists is None:
                continue

            for j in range(self._num_agents):
                if i == j:
                    continue
                pred_dist = other_action_dists[j]
                true_dist = policies[j].get_pi()
                acc = get_action_dist_distance(pred_dist, true_dist)
                self._episode_acc[i][j].append(acc)

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._episode_steps)
            for i in range(self._num_agents):
                for j, acc in self._episode_acc[i].items():
                    self._acc[i][j].append(np.mean(acc, axis=0))
                if self._track_per_step:
                    self._step_acc[i][j].append(acc)

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._acc = {}
        for i in range(self._num_agents):
            self._acc[i] = {j: [] for j in range(self._num_agents) if j != i}
            self._step_acc[i] = {
                j: [] for j in range(self._num_agents) if j != i
            }

    def reset_episode(self):
        self._episode_steps = 0
        self._episode_acc = {}
        for i in range(self._num_agents):
            self._episode_acc[i] = {
                j: [] for j in range(self._num_agents) if j != i
            }

    def get_episode(self) -> AgentStatisticsMap:
        if self._step_limit is not None:
            num_steps = self._step_limit
        else:
            num_steps = 0
            for i in range(self._num_agents):
                for acc in self._episode_acc[i].values():
                    num_steps = max(num_steps, len(acc))

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {f"action_dist_distance_{i}": np.nan}
            for j, acc in self._episode_acc[i].items():
                stats[i][f"action_dist_distance_{j}"] = np.mean(acc, axis=0)

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"action_dist_distance_{i}_{t}"] = np.nan

                for j, acc in self._episode_acc[i].items():
                    for t in range(num_steps):
                        acc_value = np.nan if len(acc) <= t else acc[t]
                        stats[i][f"action_dist_distance_{j}_{t}"] = acc_value

        return stats

    def get(self) -> AgentStatisticsMap:
        if self._step_limit is not None:
            num_steps = self._step_limit
        else:
            num_steps = 0
            for i in range(self._num_agents):
                for ep_accs in self._step_acc[i].values():
                    max_ep_len = max(len(ep) for ep in ep_accs)
                    num_steps = max(max_ep_len, num_steps)

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {f"action_dist_distance_{i}": np.nan}
            for j, acc in self._acc[i].items():
                stats[i][f"action_dist_distance_{j}"] = np.mean(acc, axis=0)

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"action_dist_distance_{i}_{t}"] = np.nan

                for j, ep_accs in self._step_acc[i].items():
                    accs_by_t = [[] for _ in range(num_steps)]
                    for acc in ep_accs:
                        for t, v in enumerate(acc):
                            accs_by_t[t].append(v)

                    for t in range(num_steps):
                        mean_acc = np.mean(accs_by_t[t])
                        stats[i][f"action_dist_distance_{j}_{t}"] = mean_acc

        return stats


def make_dir(exp_name: str) -> str:
    """Make a new experiment results directory at."""
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{exp_name}_{datetime.now()}")
    pathlib.Path(result_dir).mkdir(exist_ok=True)
    return result_dir


def compile_result_files(save_dir: str,
                         result_filepaths: List[str],
                         extra_output_dir: Optional[str] = None) -> str:
    """Compile list of results files into a single file."""
    concat_resultspath = os.path.join(save_dir, COMPILED_RESULTS_FNAME)

    dfs = list(map(pd.read_csv, result_filepaths))

    def do_concat_df(df0, df1):
        exp_ids0 = df0["exp_id"].unique().tolist()
        exp_ids1 = df1["exp_id"].unique().tolist()
        if len(set(exp_ids0).intersection(exp_ids1)) > 0:
            df1["exp_id"] += max(exp_ids0) + 1
        return pd.concat([df0, df1], ignore_index=True)

    concat_df = dfs[0]
    for df_i in dfs[1:]:
        concat_df = do_concat_df(concat_df, df_i)

    concat_df.to_csv(concat_resultspath)

    if extra_output_dir:
        extra_results_filepath = os.path.join(
            extra_output_dir, COMPILED_RESULTS_FNAME
        )
        concat_df.to_csv(extra_results_filepath)

    return concat_resultspath


def compile_results(result_dir: str,
                    extra_output_dir: Optional[str] = None) -> str:
    """Compile all .csv results files in a directory into a single file.

    If extra_output_dir is provided then will additionally compile_result to
    the extra_output_dir.

    If handle_duplicate_exp_ids is True, then function will assign new unique
    exp_ids to entries that have duplicate exp_ids.
    """
    result_filepaths = [
        os.path.join(result_dir, f) for f in os.listdir(result_dir)
        if (
            os.path.isfile(os.path.join(result_dir, f))
            and f.endswith(".csv")
            and not f.startswith(COMPILED_RESULTS_FNAME)
        )
    ]

    concat_resultspath = compile_result_files(
        result_dir, result_filepaths, extra_output_dir
    )
    return concat_resultspath


def format_as_table(values: AgentStatisticsMap) -> str:
    """Format values as a table."""
    table = PrettyTable()

    agent_ids = list(values)
    table.field_names = ["AgentID"] + [str(i) for i in agent_ids]

    for row_name in list(values[agent_ids[0]].keys()):
        row = [row_name]
        for i in agent_ids:
            agent_row_value = values[i][row_name]
            if isinstance(agent_row_value, float):
                row.append(f"{agent_row_value:.4f}")
            else:
                row.append(str(agent_row_value))
        table.add_row(row)

    table.align = 'r'
    table.align["AgentID"] = 'l'   # type: ignore
    return table.get_string()


class Writer(abc.ABC):
    """Abstract logging object for writing results to some destination.

    Each 'write()' and 'write_episode()' takes an 'OrderedDict'
    """

    @abc.abstractmethod
    def write(self, statistics: AgentStatisticsMap):
        """Write statistics to destination.."""

    @abc.abstractmethod
    def write_episode(self, statistics: AgentStatisticsMap):
        """Write episode statistics to destination.."""

    @abc.abstractmethod
    def close(self):
        """Close the Writer."""


class NullWriter(Writer):
    """Placholder Writer class that does nothing."""

    def write(self, statistics: AgentStatisticsMap):
        return

    def write_episode(self, statistics: AgentStatisticsMap):
        return

    def close(self):
        return


class CSVWriter(Writer):
    """A logging object to write to CSV files.

    Each 'write()' takes an 'OrderedDict', creating one column in the CSV file
    for each dictionary key on the first call. Subsequent calls to 'write()'
    must contain the same dictionary keys.

    Inspired by:
    https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/parts.py

    Does not support the 'write_episode()' function. Or rather it does nothing.
    """

    DEFAULT_RESULTS_FILENAME = "results.csv"

    def __init__(self,
                 filepath: Optional[str] = None,
                 dirpath: Optional[str] = None):
        if filepath is not None and dirpath is None:
            dirpath = os.path.dirname(filepath)
        elif filepath is None and dirpath is not None:
            filepath = os.path.join(dirpath, self.DEFAULT_RESULTS_FILENAME)
        else:
            raise AssertionError("Expects filepath or dirpath, not both")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        self._filepath = filepath
        self._header_written = False
        self._fieldnames: Sequence[Any] = []

    def write(self, statistics: AgentStatisticsMap):
        """Append given statistics as new rows to CSV file.

        1 row per agent entry in the AgentStatisticsMap.
        Assumes all agent's statistics maps share the same headers
        """
        agent_ids = list(statistics)
        if self._fieldnames == []:
            self._fieldnames = list(statistics[agent_ids[0]].keys())

        # Open a file in 'append' mode, so we can continue logging safely to
        # the same file if needed.
        with open(self._filepath, 'a') as fout:
            # Always use same fieldnames to create writer, this way a
            # consistency check is performed automatically on each write.
            writer = csv.DictWriter(fout, fieldnames=self._fieldnames)
            # Write a header if this is the very first write.
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            for i in agent_ids:
                writer.writerow(statistics[i])

    def write_episode(self, statistics: AgentStatisticsMap):
        return

    def close(self):
        return


class ExperimentWriter(Writer):
    """A logging object for writing results during experiments.

    This logger handles storing of results after each episode of an experiment
    as well as the final summarized results.

    The results are stored in two seperate files:
    - "exp_<exp_id>_episodes.csv": stores results for each episode
    - "exp_<exp_id>.csv": stores summary results for experiment

    Includes an additional function "checkpoint" for checkpointing results
    during experiments. This function takes a list of Tracker objects as input
    and writes a summary of the results so far to the summary results file.
    This function is useful for experiments that may take a long time to run or
    could be interupted early.

    """

    def __init__(self,
                 exp_id: int,
                 dirpath: str,
                 exp_params: AgentStatisticsMap):
        self._episode_filepath = os.path.join(
            dirpath, f"exp_{exp_id}_episodes.csv"
        )
        self._filepath = os.path.join(dirpath, f"exp_{exp_id}.csv")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        self._exp_params = exp_params

        self._episode_header_written = False
        self._episode_fieldnames: Sequence[Any] = []
        self._header_written = False
        self._fieldnames: Sequence[Any] = []

    def write_episode(self, statistics: AgentStatisticsMap):
        """Append given statistics as new rows to episode results CSV file.

        1 row per agent entry in the AgentStatisticsMap.
        Assumes all agent's statistics maps share the same headers

        Will handle adding experiment parameters to result rows.

        """
        agent_ids = list(statistics)
        statistics = combine_statistics([statistics, self._exp_params])

        if self._episode_fieldnames == []:
            self._episode_fieldnames = list(statistics[agent_ids[0]].keys())

        # Open in 'append' mode to add to results file
        with open(self._episode_filepath, 'a') as fout:
            writer = csv.DictWriter(fout, fieldnames=self._episode_fieldnames)
            if not self._episode_header_written:
                writer.writeheader()
                self._episode_header_written = True
            for i in agent_ids:
                writer.writerow(statistics[i])

    def write(self, statistics: AgentStatisticsMap):
        """Write results summary to results summary CSV file."""
        agent_ids = list(statistics)
        statistics = combine_statistics([statistics, self._exp_params])

        if self._fieldnames == []:
            self._fieldnames = list(statistics[agent_ids[0]].keys())

        # Open in 'write' mode to overwrite any previous summary results
        with open(self._filepath, 'w') as fout:
            writer = csv.DictWriter(fout, fieldnames=self._fieldnames)
            writer.writeheader()
            for i in agent_ids:
                writer.writerow(statistics[i])

    def close(self):
        """Close the `ExperimentWriter`."""

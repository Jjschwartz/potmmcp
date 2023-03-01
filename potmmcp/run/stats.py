import abc
import time
from collections import ChainMap
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import posggym
import posggym.model as M
from scipy.stats import wasserstein_distance

import potmmcp.policy as P
import potmmcp.tree as tree_lib


AgentStatisticsMap = Dict[M.AgentID, Dict[str, Any]]


def generate_episode_statistics(trackers: Iterable["Tracker"]) -> AgentStatisticsMap:
    """Generate episode statistics from set of trackers."""
    statistics = combine_statistics([t.get_episode() for t in trackers])
    return statistics


def generate_statistics(trackers: Iterable["Tracker"]) -> AgentStatisticsMap:
    """Generate summary statistics from set of trackers."""
    statistics = combine_statistics([t.get() for t in trackers])
    return statistics


def combine_statistics(
    statistic_maps: Sequence[AgentStatisticsMap],
) -> AgentStatisticsMap:
    """Combine multiple Agent statistic maps into a single one."""
    agent_ids = list(statistic_maps[0].keys())
    return {
        i: dict(ChainMap(*(stat_maps[i] for stat_maps in statistic_maps)))
        for i in agent_ids
    }


def get_action_dist_distance(dist1: P.ActionDist, dist2: P.ActionDist) -> float:
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

    return wasserstein_distance(actions, actions, probs1, probs2)


def get_default_trackers(
    num_agents: int, discounts: Union[List[float], float]
) -> List["Tracker"]:
    """Get the default set of Trackers."""
    return [
        EpisodeTracker(num_agents, discounts),
        SearchTimeTracker(num_agents),
    ]


def belief_tracker_fn(kwargs) -> Sequence["Tracker"]:
    """Get trackers for POTMMCP experiment that track beliefs + defaults.

    Only includes tracking of Bayes accuracy (i.e. belief accuracy of other
    agent policy) and Action Distribution Distance.

    Doesn't track history or state belief accuracy since these can be expensive
    to compute and typically have low accuracy by default due to their
    combinatorial nature.

    Required kwargs
    ---------------
    num_agents : int
    discount : float
    step_limit : int

    """
    num_agents = kwargs["num_agents"]
    discount = kwargs["discount"]

    trackers = get_default_trackers(num_agents, discount)

    tracker_kwargs = {
        "num_agents": num_agents,
        # only track per step if step limit is provided
        "track_per_step": kwargs["step_limit"] is not None,
        "step_limit": kwargs["step_limit"],
    }
    trackers.append(BayesAccuracyTracker(**tracker_kwargs))
    trackers.append(ActionDistributionDistanceTracker(**tracker_kwargs))
    return trackers


class Tracker(abc.ABC):
    """Generic Tracker Base class."""

    @abc.abstractmethod
    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ):
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

    def __init__(self, num_agents: int, discounts: Union[float, List[float]]):
        if isinstance(discounts, float):
            discounts = [discounts] * num_agents
        assert len(discounts) == num_agents

        self._num_agents = num_agents
        self._discounts = np.array(discounts)

        self._num_episodes = 0
        self._current_episode_done = False
        self._current_episode_start_time = time.time()
        self._current_episode_returns = np.zeros(num_agents)
        self._current_episode_discounted_returns = np.zeros(num_agents)
        self._current_episode_steps = 0

        self._dones = []  # type: ignore
        self._times = []  # type: ignore
        self._returns = []  # type: ignore
        self._discounted_returns = []  # type: ignore
        self._steps = []  # type: ignore
        self._outcomes = []  # type: ignore

    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ):
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
            self._discounted_returns.append(self._current_episode_discounted_returns)
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
                "episode_discounted_return": (self._discounted_returns[-1][i]),
                "episode_steps": self._steps[-1],
                "episode_outcome": self._outcomes[-1][i],
                "episode_done": self._dones[-1],
                "episode_time": self._times[-1],
            }
        return stats

    def get(self) -> AgentStatisticsMap:
        outcome_counts = {k: [0 for _ in range(self._num_agents)] for k in M.Outcome}
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
                "episode_discounted_return_mean": (
                    np.mean(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_return_std": (
                    np.std(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_return_max": (
                    np.max(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_return_min": (
                    np.min(self._discounted_returns, axis=0)[i]
                ),
                "episode_steps_mean": np.mean(self._steps),
                "episode_steps_std": np.std(self._steps),
                "episode_time_mean": np.mean(self._times),
                "episode_time_std": np.std(self._times),
                "num_episode_done": np.sum(self._dones),
            }

            for outcome, counts in outcome_counts.items():
                stats[i][f"num_{outcome}"] = counts[i]

        return stats


class SearchTimeTracker(Tracker):
    """Tracks Search, Update, Reinvigoration time in Search Trees."""

    # The list of keys to track from the policies.statistics property
    TIME_KEYS = [
        "search_time",
        "update_time",
        "reinvigoration_time",
        "evaluation_time",
        "policy_calls",
        "inference_time",
        "search_depth",
        "num_sims",
        "min_value",
        "max_value",
    ]

    def __init__(self, num_agents: int):
        self._num_agents = num_agents

        self._num_episodes = 0
        self._current_episode_steps = 0
        self._current_episode_times: Dict[M.AgentID, Dict[str, List[float]]] = {}

        self._steps: List[int] = []
        self._times: Dict[M.AgentID, Dict[str, List[float]]] = {}

        self.reset()

    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ):
        if episode_t == 0:
            return

        self._current_episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.POTMMCP):
                continue

            statistics = policies[i].statistics
            for time_key in self.TIME_KEYS:
                self._current_episode_times[i][time_key].append(
                    statistics.get(time_key, np.nan)
                )

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._current_episode_steps)
            for i in range(self._num_agents):
                if not isinstance(policies[i], tree_lib.POTMMCP):
                    for k in self.TIME_KEYS:
                        self._times[i][k].append(np.nan)
                else:
                    for k in self.TIME_KEYS:
                        key_step_times = self._current_episode_times[i][k]
                        self._times[i][k].append(np.nanmean(key_step_times))

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
                if len(step_times) == 0 or np.isnan(np.sum(step_times)):
                    mean_val = np.nan
                else:
                    mean_val = np.nanmean(step_times, axis=0)
                agent_stats[key] = mean_val
            stats[i] = agent_stats

        return stats

    def get(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            agent_stats = {}
            for key, values in self._times[i].items():
                if len(values) == 0 or np.isnan(np.sum(values)):
                    mean_val = np.nan
                    std_val = np.nan
                else:
                    mean_val = np.nanmean(values, axis=0)
                    std_val = np.nanstd(values, axis=0)
                agent_stats[f"{key}_mean"] = mean_val
                agent_stats[f"{key}_std"] = std_val
            stats[i] = agent_stats
        return stats


class BayesAccuracyTracker(Tracker):
    """Tracks Accuracy between distribution over pis and the true pi.

    Only tracks for POTMMCP policy and if the opponent policy has a policy ID
    that matches a policy ID within the POTMMCP other agent policy
    distribution.

    If track_per_step=True, then outputs accuracy for each episode step in the
    for "bayes_accuracy_<agent_id>_<step_num>". This may generate a lot of
    output, so beware.

    """

    def __init__(
        self, num_agents: int, track_per_step: bool, step_limit: Optional[int] = None
    ):
        self._num_agents = num_agents
        self._track_per_step = track_per_step
        self._step_limit = step_limit

        self._num_episodes = 0
        self._episode_steps = 0
        self._episode_acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}

        self._steps: List[int] = []
        self._acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}
        self._step_acc: Dict[M.AgentID, Dict[M.AgentID, List[List[float]]]] = {}

        self.reset()

    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ):
        if episode_t == 0:
            return

        self._episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.POTMMCP):
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
                    if i == j or len(acc) == 0:
                        mean_acc = np.nan
                    else:
                        mean_acc = np.nanmean(acc, axis=0)
                    self._acc[i][j].append(mean_acc)
                    if self._track_per_step:
                        self._step_acc[i][j].append(acc)

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._acc = {}
        for i in range(self._num_agents):
            self._acc[i] = {j: [] for j in range(self._num_agents) if j != i}
            self._step_acc[i] = {j: [] for j in range(self._num_agents) if j != i}

    def reset_episode(self):
        self._episode_steps = 0
        self._episode_acc = {}
        for i in range(self._num_agents):
            self._episode_acc[i] = {j: [] for j in range(self._num_agents) if j != i}

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
            stats[i] = {
                f"bayes_accuracy_{i}_mean": np.nan,
                f"bayes_accuracy_{i}_std": np.nan,
            }
            for j, acc in self._episode_acc[i].items():
                if i == j or len(acc) == 0:
                    mean_acc = np.nan
                    std_acc = np.nan
                else:
                    mean_acc = np.nanmean(acc, axis=0)
                    std_acc = np.nanstd(acc, axis=0)
                stats[i][f"bayes_accuracy_{j}_mean"] = mean_acc
                stats[i][f"bayes_accuracy_{j}_std"] = std_acc

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
            stats[i] = {
                f"bayes_accuracy_{i}_mean": np.nan,
                f"bayes_accuracy_{i}_std": np.nan,
            }
            for j, acc in self._acc[i].items():
                if i == j or len(acc) == 0:
                    mean_acc = np.nan
                    std_acc = np.nan
                else:
                    mean_acc = np.nanmean(acc, axis=0)
                    std_acc = np.nanstd(acc, axis=0)
                stats[i][f"bayes_accuracy_{j}_mean"] = mean_acc
                stats[i][f"bayes_accuracy_{j}_std"] = std_acc

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"bayes_accuracy_{i}_{t}_mean"] = np.nan
                    stats[i][f"bayes_accuracy_{i}_{t}_std"] = np.nan

                for j, ep_accs in self._step_acc[i].items():
                    accs_by_t = [[] for _ in range(num_steps)]
                    for acc in ep_accs:
                        for t, v in enumerate(acc):
                            accs_by_t[t].append(v)

                    for t in range(num_steps):
                        if i == j or len(acc) == 0:
                            mean_acc = np.nan
                            std_acc = np.nan
                        else:
                            mean_acc = np.nanmean(accs_by_t[t])
                            std_acc = np.nanstd(accs_by_t[t])
                        stats[i][f"bayes_accuracy_{j}_{t}_mean"] = mean_acc
                        stats[i][f"bayes_accuracy_{j}_{t}_std"] = std_acc
        return stats


class BeliefStateAccuracyTracker(Tracker):
    """Tracks accuracy between belief over states and the true state.

    Only tracks for POTMMCP policy

    If track_per_step=True, then outputs accuracy for each episode step in the
    for "belief_accuracy_<agent_id>_<step_num>". This may generate a lot of
    output, so beware.

    """

    def __init__(
        self, num_agents: int, track_per_step: bool, step_limit: Optional[int] = None
    ):
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

    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ):
        if episode_t == 0:
            self._prev_state = env.state
            return

        self._episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.POTMMCP):
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
                self._acc[i].append(np.nanmean(self._episode_acc[i], axis=0))
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
            stats[i] = {
                "state_accuracy_mean": np.nanmean(acc, axis=0),
                "state_accuracy_std": np.nanstd(acc, axis=0),
            }

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
            stats[i] = {
                "state_accuracy_mean": np.nanmean(self._acc[i], axis=0),
                "state_accuracy_std": np.nanstd(self._acc[i], axis=0),
            }

            if self._track_per_step:
                ep_accs = self._step_acc[i]
                accs_by_t = [[] for _ in range(num_steps)]
                for acc in ep_accs:
                    for t, v in enumerate(acc):
                        accs_by_t[t].append(v)

                for t in range(num_steps):
                    stats[i][f"state_accuracy_{t}_mean"] = np.nanmean(accs_by_t[t])
                    stats[i][f"state_accuracy_{t}_std"] = np.nanstd(accs_by_t[t])

        return stats


class BeliefHistoryAccuracyTracker(Tracker):
    """Tracks accuracy of the belief over history of the other agents.

    Only tracks for POTMMCP policy

    If track_per_step=True, then outputs accuracy for each episode step in the
    for "history_accuracy_<agent_id>_<step_num>". This may generate a lot of
    output, so beware.

    """

    def __init__(
        self, num_agents: int, track_per_step: bool, step_limit: Optional[int] = None
    ):
        self._num_agents = num_agents
        self._track_per_step = track_per_step
        self._step_limit = step_limit

        self._num_episodes = 0
        self._episode_steps = 0
        self._episode_acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}

        self._steps: List[int] = []
        self._acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}
        self._step_acc: Dict[M.AgentID, Dict[M.AgentID, List[List[float]]]] = {}

        self.reset()

    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ):
        if episode_t == 0:
            return

        self._episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.POTMMCP):
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
                    self._acc[i][j].append(np.nanmean(acc, axis=0))
                if self._track_per_step:
                    self._step_acc[i][j].append(acc)

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._acc = {}
        for i in range(self._num_agents):
            self._acc[i] = {j: [] for j in range(self._num_agents) if j != i}
            self._step_acc[i] = {j: [] for j in range(self._num_agents) if j != i}

    def reset_episode(self):
        self._episode_steps = 0
        self._episode_acc = {}
        for i in range(self._num_agents):
            self._episode_acc[i] = {j: [] for j in range(self._num_agents) if j != i}

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
            stats[i] = {
                f"history_accuracy_{i}_mean": np.nan,
                f"history_accuracy_{i}_std": np.nan,
            }
            for j, acc in self._acc[i].items():
                stats[i][f"history_accuracy_{j}_mean"] = np.nanmean(acc, axis=0)
                stats[i][f"history_accuracy_{j}_std"] = np.nanstd(acc, axis=0)

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
            stats[i] = {
                f"history_accuracy_{i}_mean": np.nan,
                f"history_accuracy_{i}_std": np.nan,
            }
            for j, acc in self._acc[i].items():
                stats[i][f"history_accuracy_{j}_mean"] = np.nanmean(acc, axis=0)
                stats[i][f"history_accuracy_{j}_std"] = np.nanstd(acc, axis=0)

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"history_accuracy_{i}_{t}_mean"] = np.nan
                    stats[i][f"history_accuracy_{i}_{t}_std"] = np.nan

                for j, ep_accs in self._step_acc[i].items():
                    accs_by_t = [[] for _ in range(num_steps)]
                    for acc in ep_accs:
                        for t, v in enumerate(acc):
                            accs_by_t[t].append(v)

                    for t in range(num_steps):
                        mean_acc = np.nanmean(accs_by_t[t])
                        stats[i][f"history_accuracy_{j}_{t}_mean"] = mean_acc
                        std_acc = np.nanstd(accs_by_t[t])
                        stats[i][f"history_accuracy_{j}_{t}_std"] = std_acc

        return stats


class ActionDistributionDistanceTracker(Tracker):
    """Tracks distance between expected and true policy of the other agents.

    Only tracks for POTMMCP policy.

    This looks specifically at the expected distribution over actions of the
    other agent at the current root node of the POTMMCP policy tree. This is
    an expectation over the history of the other agent and the policy of the
    other agent for the root belief.

    If track_per_step=True, then outputs accuracy for each episode step in the
    for "history_accuracy_<agent_id>_<step_num>". This may generate a lot of
    output, so beware.

    Note this reports the Wassersteing Distance between the expected policy and
    the true agent policy.

    """

    def __init__(
        self, num_agents: int, track_per_step: bool, step_limit: Optional[int] = None
    ):
        self._num_agents = num_agents
        self._track_per_step = track_per_step
        self._step_limit = step_limit

        self._num_episodes = 0
        self._episode_steps = 0
        self._episode_acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}

        self._steps: List[int] = []
        self._acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}
        self._step_acc: Dict[M.AgentID, Dict[M.AgentID, List[List[float]]]] = {}

        self.reset()

    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ):
        if episode_t == 0:
            return

        self._episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.POTMMCP):
                continue

            other_action_dists = tree_lib.get_other_agent_action_dist(policies[i])
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
                    if i == j or len(acc) == 0:
                        mean_acc = np.nan
                    else:
                        mean_acc = np.nanmean(acc, axis=0)
                    self._acc[i][j].append(mean_acc)
                    if self._track_per_step:
                        self._step_acc[i][j].append(acc)

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._acc = {}
        for i in range(self._num_agents):
            self._acc[i] = {j: [] for j in range(self._num_agents) if j != i}
            self._step_acc[i] = {j: [] for j in range(self._num_agents) if j != i}

    def reset_episode(self):
        self._episode_steps = 0
        self._episode_acc = {}
        for i in range(self._num_agents):
            self._episode_acc[i] = {j: [] for j in range(self._num_agents) if j != i}

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
            stats[i] = {
                f"action_dist_distance_{i}_mean": np.nan,
                f"action_dist_distance_{i}_std": np.nan,
            }
            for j, acc in self._episode_acc[i].items():
                if i == j or len(acc) == 0:
                    mean_acc = np.nan
                    std_acc = np.nan
                else:
                    mean_acc = np.nanmean(acc, axis=0)
                    std_acc = np.nanstd(acc, axis=0)
                stats[i][f"action_dist_distance_{j}_mean"] = mean_acc
                stats[i][f"action_dist_distance_{j}_std"] = std_acc

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
            stats[i] = {
                f"action_dist_distance_{i}_mean": np.nan,
                f"action_dist_distance_{i}_std": np.nan,
            }
            for j, acc in self._acc[i].items():
                if i == j or len(acc) == 0:
                    mean_acc = np.nan
                    std_acc = np.nan
                else:
                    mean_acc = np.nanmean(acc, axis=0)
                    std_acc = np.nanstd(acc, axis=0)
                stats[i][f"action_dist_distance_{j}_mean"] = mean_acc
                stats[i][f"action_dist_distance_{j}_std"] = std_acc

            if self._track_per_step:
                for t in range(num_steps):
                    stats[i][f"action_dist_distance_{i}_{t}_mean"] = np.nan
                    stats[i][f"action_dist_distance_{i}_{t}_std"] = np.nan

                for j, ep_accs in self._step_acc[i].items():
                    accs_by_t = [[] for _ in range(num_steps)]
                    for acc in ep_accs:
                        for t, v in enumerate(acc):
                            accs_by_t[t].append(v)

                    for t in range(num_steps):
                        if i == j or len(accs_by_t) == 0:
                            mean_acc = np.nan
                            std_acc = np.nan
                        else:
                            mean_acc = np.nanmean(accs_by_t[t])
                            std_acc = np.nanstd(accs_by_t[t])
                        stats[i][f"action_dist_distance_{j}_{t}_mean"] = mean_acc
                        stats[i][f"action_dist_distance_{j}_{t}_std"] = std_acc
        return stats

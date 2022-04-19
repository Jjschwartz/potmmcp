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
from prettytable import PrettyTable

import posggym
import posggym.model as M

import baposgmcp.tree as tree_lib
import baposgmcp.policy as policy_lib
from baposgmcp.config import BASE_RESULTS_DIR

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
             episode_end: bool) -> None:
        """Accumulates statistics for a single step."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all gathered statistics."""

    @abc.abstractmethod
    def reset_episode(self) -> None:
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
             episode_end: bool) -> None:
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

    def reset(self) -> None:
        self.reset_episode()
        self._num_episodes = 0
        self._dones = []
        self._times = []
        self._returns = []
        self._discounted_returns = []
        self._steps = []
        self._outcomes = []

    def reset_episode(self) -> None:
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
             episode_end: bool) -> None:
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

    def reset(self) -> None:
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._times = {}
        for i in range(self._num_agents):
            self._times[i] = {k: [] for k in self.TIME_KEYS}

    def reset_episode(self) -> None:
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
    """

    def __init__(self, num_agents: int):
        self._num_agents = num_agents

        self._num_episodes = 0
        self._current_episode_steps = 0
        self._current_episode_acc: Dict[
            M.AgentID, Dict[M.AgentID, List[float]]
        ] = {}

        self._steps: List[int] = []
        self._acc: Dict[M.AgentID, Dict[M.AgentID, List[float]]] = {}

        self.reset()

    def step(self,
             episode_t: int,
             env: posggym.Env,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy_lib.BasePolicy],
             episode_end: bool) -> None:
        if episode_t == 0:
            return

        self._current_episode_steps += 1

        for i in range(self._num_agents):
            if not isinstance(policies[i], tree_lib.BAPOSGMCP):
                continue

            pi_beliefs = tree_lib.get_other_pis_belief(policies[i])
            for j in range(self._num_agents):
                if i == j:
                    continue
                policy_id_j = policies[j].policy_id
                acc = self._calculate_acc(pi_beliefs[j], policy_id_j)
                self._current_episode_acc[i][j].append(acc)

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._current_episode_steps)
            for i in range(self._num_agents):
                for j, acc in self._current_episode_acc[i].items():
                    self._acc[i][j].append(np.mean(acc, axis=0))

    def _calculate_acc(self, policy_dist, true_policy) -> float:
        if true_policy not in policy_dist:
            return 0.0
        return policy_dist[true_policy]

    def reset(self) -> None:
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._acc = {}
        for i in range(self._num_agents):
            self._acc[i] = {j: [] for j in range(self._num_agents) if j != i}

    def reset_episode(self) -> None:
        self._current_episode_steps = 0
        self._current_episode_acc = {}
        for i in range(self._num_agents):
            self._current_episode_acc[i] = {
                j: [] for j in range(self._num_agents) if j != i
            }

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            stats[i] = {
                f"bayes_accuracy_{i}": np.nan
            }
            for j, acc in self._current_episode_acc[i].items():
                stats[i][f"bayes_accuracy_{j}"] = np.mean(acc, axis=0)
        return stats

    def get(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            stats[i] = {
                f"bayes_accuracy_{i}": np.nan
            }
            for j, acc in self._acc[i].items():
                stats[i][f"bayes_accuracy_{j}"] = np.mean(acc, axis=0)
        return stats


def make_dir(exp_name: str) -> str:
    """Make a new experiment results directory at."""
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{exp_name}_{datetime.now()}")
    pathlib.Path(result_dir).mkdir(exist_ok=True)
    return result_dir


def compile_results(result_dir: str,
                    extra_output_dir: Optional[str] = None) -> str:
    """Compile all .tsv results files in a directory into a single file.

    If extra_output_dir is provided then will additionally compile_result to
    the extra_output_dir.
    """
    result_filepaths = [
        os.path.join(result_dir, f) for f in os.listdir(result_dir)
        if os.path.isfile(os.path.join(result_dir, f)) and f.endswith(".csv")
    ]

    concat_results_filepath = os.path.join(result_dir, "compiled_results.csv")

    dfs = map(pd.read_csv, result_filepaths)
    concat_df = pd.concat(dfs)
    concat_df.to_csv(concat_results_filepath)

    if extra_output_dir:
        extra_results_filepath = os.path.join(
            extra_output_dir, "compiled_results.csv"
        )
        concat_df.to_csv(extra_results_filepath)

    return concat_results_filepath


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

    Each 'write()' takes an 'OrderedDict'
    """

    @abc.abstractmethod
    def write(self, statistics: AgentStatisticsMap) -> None:
        """Write statistics to destination.."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the Writer."""


class NullWriter(Writer):
    """Placholder Writer class that does nothing."""

    def write(self, statistics: AgentStatisticsMap) -> None:
        return

    def close(self) -> None:
        return


class CSVWriter(Writer):
    """A logging object to write to CSV files.

    Each 'write()' takes an 'OrderedDict', creating one column in the CSV file
    for each dictionary key on the first call. Subsequent calls to 'write()'
    must contain the same dictionary keys.

    Inspired by:
    https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/parts.py
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

    def write(self, statistics: AgentStatisticsMap) -> None:
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

    def close(self) -> None:
        """Close the `CsvWriter`."""

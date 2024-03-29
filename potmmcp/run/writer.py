import abc
import csv
import json
import math
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from prettytable import PrettyTable

from potmmcp.run.stats import AgentStatisticsMap, combine_statistics


COMPILED_RESULTS_FNAME = "compiled_results.csv"


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

    table.align = "r"
    table.align["AgentID"] = "l"  # type: ignore
    return table.get_string()


def _do_concat_df(df0, df1):
    exp_ids0 = df0["exp_id"].unique().tolist()
    exp_ids1 = df1["exp_id"].unique().tolist()
    if len(set(exp_ids0).intersection(exp_ids1)) > 0:
        df1["exp_id"] += max(exp_ids0) + 1
    return pd.concat([df0, df1], ignore_index=True)


def _read_and_concat(concat_df, filepath):
    df_new = pd.read_csv(filepath)
    return _do_concat_df(concat_df, df_new)


def _read_and_concat_multiple_files(filepaths):
    num_files = len(filepaths)
    main_df = pd.read_csv(filepaths[0])
    for i, p in enumerate(filepaths[1:]):
        main_df = _read_and_concat(main_df, p)
        if num_files > 10 and (i > 0 and (i + 1) % (num_files // 10) == 0):
            print(f"[pid={os.getpid()}] {i+1}/{num_files} processed")
    return main_df


def compile_result_files(
    result_filepaths: List[str], verbose: bool = True, n_procs: int = 1
) -> pd.DataFrame:
    """Compile list of results files into a single pandas dataframe."""
    num_files = len(result_filepaths)
    if verbose:
        print(f"Loading and concatting {num_files} files")

    if n_procs == 1:
        concat_df = _read_and_concat_multiple_files(result_filepaths)
    else:
        if verbose:
            print(f"Compiling using {n_procs}")

        chunk_size = math.ceil(num_files / n_procs)
        chunks = [
            result_filepaths[i * chunk_size: (i + 1) * chunk_size]
            for i in range(n_procs)
            if len(result_filepaths[i * chunk_size: (i + 1) * chunk_size])
        ]
        with mp.Pool(n_procs) as pool:
            chunk_dfs = pool.map(_read_and_concat_multiple_files, chunks)

        if verbose:
            print("Concatting chunks")

        concat_df = chunk_dfs[0]
        for df in chunk_dfs[1:]:
            concat_df = _do_concat_df(concat_df, df)

    return concat_df


def compile_and_save_result_files(
    save_dir: str,
    result_filepaths: List[str],
    extra_output_dir: Optional[str] = None,
    compiled_results_filename: Optional[str] = None,
    verbose: bool = True,
    n_procs: int = 1,
) -> str:
    """Compile list of results files into a single file."""
    if not compiled_results_filename:
        compiled_results_filename = COMPILED_RESULTS_FNAME
    concat_resultspath = os.path.join(save_dir, compiled_results_filename)

    num_files = len(result_filepaths)
    if verbose:
        print(f"Loading and concatting {num_files} files")

    concat_df = compile_result_files(result_filepaths, verbose=verbose, n_procs=n_procs)
    concat_df.to_csv(concat_resultspath, index=False)

    if extra_output_dir:
        extra_results_filepath = os.path.join(extra_output_dir, COMPILED_RESULTS_FNAME)
        concat_df.to_csv(extra_results_filepath, index=False)

    return concat_resultspath


def compile_results(result_dir: str, extra_output_dir: Optional[str] = None) -> str:
    """Compile all .csv results files in a directory into a single file.

    If extra_output_dir is provided then will additionally compile_result to
    the extra_output_dir.

    If handle_duplicate_exp_ids is True, then function will assign new unique
    exp_ids to entries that have duplicate exp_ids.
    """
    result_filepaths = [
        os.path.join(result_dir, f)
        for f in os.listdir(result_dir)
        if (
            os.path.isfile(os.path.join(result_dir, f))
            and ExperimentWriter.is_results_file(f)
            and not f.startswith(COMPILED_RESULTS_FNAME)
        )
    ]

    concat_resultspath = compile_and_save_result_files(
        save_dir=result_dir,
        result_filepaths=result_filepaths,
        extra_output_dir=extra_output_dir,
        verbose=True,
        n_procs=1,
    )
    return concat_resultspath


def write_dict(args: Dict[str, Any], output_file: str):
    """Write dictionary to file."""
    args = _convert_keys_to_str(args)
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(args, fout)
    return output_file


def _convert_keys_to_str(x: Dict) -> Dict:
    y = {}
    for k, v in x.items():
        if isinstance(v, dict):
            v = _convert_keys_to_str(v)
        y[str(k)] = v
    return y


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

    def __init__(self, filepath: Optional[str] = None, dirpath: Optional[str] = None):
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
        with open(self._filepath, "a") as fout:
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

    def __init__(self, exp_id: int, dirpath: str, exp_params: AgentStatisticsMap):
        self._episode_filepath = os.path.join(dirpath, f"exp_{exp_id}_episodes.csv")
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
        with open(self._episode_filepath, "a") as fout:
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
        with open(self._filepath, "w") as fout:
            writer = csv.DictWriter(fout, fieldnames=self._fieldnames)
            writer.writeheader()
            for i in agent_ids:
                writer.writerow(statistics[i])

    def close(self):
        """Close the `ExperimentWriter`."""

    @staticmethod
    def is_results_file(filename: str) -> bool:
        """Check if filename is for an experiment summary results file."""
        if not filename.endswith(".csv"):
            return False
        filename = filename.replace(".csv", "")
        tokens = filename.split("_")
        if len(tokens) != 2 or tokens[0] != "exp":
            return False
        try:
            int(tokens[1])
            return True
        except ValueError:
            return False

    @staticmethod
    def is_episodes_results_file(filename: str) -> bool:
        """Check if filename is for an experiment episode results file."""
        if not filename.endswith(".csv"):
            return False
        filename = filename.replace(".csv", "")
        tokens = filename.split("_")
        if len(tokens) != 3 or tokens[0] != "exp" or tokens[2] != "episodes":
            return False
        try:
            int(tokens[1])
            return True
        except ValueError:
            return False

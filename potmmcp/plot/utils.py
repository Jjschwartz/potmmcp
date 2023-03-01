"""Plotting utility functions."""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import posggym


def add_95CI(df: pd.DataFrame) -> pd.DataFrame:
    """Add 95% CI columns to dataframe."""

    def conf_int(row, prefix):
        std = row[f"{prefix}_std"]
        if f"{prefix}_n" in row:
            # belief stat
            n = row[f"{prefix}_n"]
        else:
            n = row["num_episodes"]
        return 1.96 * (std / np.sqrt(n))

    prefix = ""
    for col in df.columns:
        if not col.endswith("_std"):
            continue
        prefix = col.replace("_std", "")
        df[f"{prefix}_CI"] = df.apply(lambda row: conf_int(row, prefix), axis=1)
    return df


def add_outcome_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Add proportion columns to dataframe."""

    def prop(row, col_name):
        n = row["num_episodes"]
        total = row[col_name]
        return total / n

    columns = ["num_LOSS", "num_DRAW", "num_WIN", "num_NA"]
    new_column_names = ["prop_LOSS", "prop_DRAW", "prop_WIN", "prop_NA"]
    for col_name, new_name in zip(columns, new_column_names):
        if col_name in df.columns:
            df[new_name] = df.apply(lambda row: prop(row, col_name), axis=1)
    return df


def clean_df_policy_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Remove environment name from policy ID, if it's present."""

    def clean(row):
        if "/" in row["policy_id"]:
            return row["policy_id"].split("/")[1]
        return row["policy_id"]

    df["policy_id"] = df.apply(clean, axis=1)
    return df


def add_df_coplayer_policy_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add co-player policy IDs and co-team ID to dataframe .

    Adds a new column for each agent in the environment:

      coplayer_policy_id_0, coplayer_policy_id_1, ..., coplayer_policy_id_N

    Each column contains the policy_id of the agent with corresponding ID for
    the given experiment.

    This includes the row agent so if the row["agent_id"] = i
    then row["coplayer_policy_id_i"] = row["policy_id"]

    Also add "co_team_id" column to environment which is the tuple of co-player policy
    IDs (excluding the row agent).

    """
    agent_ids = df["agent_id"].unique().tolist()
    agent_ids.sort()

    dfs = [df[df["agent_id"] == i] for i in agent_ids]
    for i, df_i in zip(agent_ids, dfs):
        for j, df_j in zip(agent_ids, dfs):
            df_i[f"coplayer_policy_id_{j}"] = df_i["exp_id"].map(
                df_j.set_index("exp_id")["policy_id"].to_dict()
            )
    df = pd.concat(dfs).reset_index(drop=True)

    def get_team_id(row):
        i = row["agent_id"]
        pi_ids = [row[f"coplayer_policy_id_{j}"] for j in agent_ids if j != i]
        return tuple(pi_ids)

    df["co_team_id"] = df.apply(get_team_id, axis=1)
    return df


def clean_num_sims(df):
    """Get num sims in integer format."""

    def clean(row):
        try:
            return int(row["num_sims"])
        except (KeyError, ValueError, TypeError):
            pass
        try:
            return int(row["belief_size"])
        except (KeyError, ValueError, TypeError):
            pass
        return 0

    df["num_sims"] = df.apply(clean, axis=1)
    df = df.astype({"num_sims": int})
    return df


def clean_search_time_limit(df):
    """Get search time limit in float format."""

    def clean(row):
        try:
            return float(row["search_time_limit"])
        except (KeyError, ValueError, TypeError):
            pass
        return np.nan

    df["search_time_limit"] = df.apply(clean, axis=1)
    df = df.astype({"search_time_limit": float})
    return df


def clean_truncated(df):
    """Get truncated in bool format.

    If not applicable (i.e. for non-MCTS based policies) then sets to False.
    """

    def clean(row):
        if "truncated" not in row:
            return False

        v = row["truncated"]
        if isinstance(v, bool):
            return v
        if v == "True":
            return True
        if v in ("False", None, "None"):
            return False
        Warning(f"Value '{v}' for truncated param is confusing. Setting to False")
        return False

    df["truncated"] = df.apply(clean, axis=1)
    df = df.astype({"truncated": bool})
    return df


def import_results(
    result_file: str,
    columns_to_drop: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Import experiment results."""
    # disable annoying warning
    pd.options.mode.chained_assignment = None

    df = pd.read_csv(result_file)
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1, errors="ignore")

    # rename
    df.rename(
        columns={
            "num_outcome_LOSS": "num_LOSS",
            "num_outcome_DRAW": "num_DRAW",
            "num_outcome_WIN": "num_WIN",
            "num_outcome_NA": "num_NA",
        }
    )

    df = add_95CI(df)
    df = add_outcome_proportions(df)
    df = clean_num_sims(df)
    df = clean_search_time_limit(df)
    df = clean_truncated(df)
    df = clean_df_policy_ids(df)
    df = add_df_coplayer_policy_ids(df)

    # enable annoyin warning
    pd.options.mode.chained_assignment = "warn"
    return df


def plot_environment(env_id: str, figsize=None):
    """Display rendering of the environment."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Turn off x/y axis numbering/ticks
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    env = posggym.make(env_id)
    env_img = env.render(mode="rgb_array")

    if isinstance(env_img, tuple):
        # render returns img for each agent
        # env img is 0th by default
        env_img = env_img[0]

    imshow_obj = ax.imshow(env_img, interpolation="bilinear", origin="upper")
    imshow_obj.set_data(env_img)

    env.close()

    return fig, ax

from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from posggym_agents.plot.heatmap import plot_pairwise_heatmap


def get_pairwise_values(plot_df,
                        y_key: str,
                        policy_key: str = "policy_id",
                        coplayer_policy_key: str = "coplayer_policy_id",
                        coplayer_policies: Optional[List[str]] = None,
                        average_duplicates: bool = True,
                        duplicate_warning: bool = False):
    """Get values for each policy pairing."""
    policies = plot_df[policy_key].unique().tolist()
    policies.sort()

    if coplayer_policies is None:
        coplayer_policies = plot_df[coplayer_policy_key].unique().tolist()
        coplayer_policies.sort()

    agent_ids = plot_df["agent_id"].unique()
    agent_ids.sort()

    plot_df = plot_df[plot_df[coplayer_policy_key].isin(coplayer_policies)]
    gb = plot_df.groupby([policy_key, coplayer_policy_key])
    pw_values = np.full((len(policies), len(coplayer_policies)), np.nan)
    for name, group in gb:
        (row_policy, col_policy) = name
        row_policy_idx = policies.index(row_policy)
        col_policy_idx = coplayer_policies.index(col_policy)

        pw_values[row_policy_idx][col_policy_idx] = group.mean()[y_key]

    return pw_values, (policies, coplayer_policies)


def plot_pairwise_comparison(plot_df,
                             y_key: str,
                             policy_key: str = "policy_id",
                             coplayer_policy_key: str = "coplayer_policy_id",
                             y_err_key: Optional[str] = None,
                             vrange=None,
                             figsize=(20, 20),
                             valfmt=None,
                             coplayer_policies: Optional[List[str]] = None,
                             average_duplicates: bool = True,
                             duplicate_warning: bool = False):
    """Plot results for each policy pairings.

    This produces a policy X policy grid-plot

    If `y_err_key` is provided then an additional policy X policy grid-plot is
    produced displaying the err values.

    It is possible that the there are multiple pairings of the same
    policy matchup.
    E.g. (agent 0 pi_0 vs agent 1 pi_1) and (agent 0 pi_1 vs agent 1 pi_0).
    In some cases we can just take the average of the two (e.g. when looking
    at times or returns). In such cases use `average_duplicates=True`.
    For cases where we can't take the average (e.g. for getting exp_id), set
    `average_duplicates=False`, in which case the first entry will be used.
    """
    if duplicate_warning:
        if average_duplicates:
            print("Averaging duplicates. FYI")
        else:
            print("Not averaging duplicates. FYI.")

    if valfmt is None:
        valfmt = "{x:.2f}"

    ncols = 2 if y_err_key else 1
    fig, axs = plt.subplots(
        nrows=1, ncols=ncols, figsize=figsize, squeeze=False, sharey=True
    )

    pw_values, (row_policies, col_policies) = get_pairwise_values(
        plot_df,
        y_key,
        policy_key=policy_key,
        coplayer_policy_key=coplayer_policy_key,
        coplayer_policies=coplayer_policies,
        average_duplicates=average_duplicates,
        duplicate_warning=duplicate_warning
    )

    plot_pairwise_heatmap(
        axs[0][0],
        (row_policies, col_policies),
        pw_values,
        title=None,
        vrange=vrange,
        valfmt=valfmt
    )

    if y_err_key:
        pw_err_values, _ = get_pairwise_values(
            plot_df,
            y_err_key,
            policy_key=policy_key,
            coplayer_policy_key=coplayer_policy_key,
            coplayer_policies=coplayer_policies,
            average_duplicates=average_duplicates,
            duplicate_warning=duplicate_warning
        )

        plot_pairwise_heatmap(
            axs[0][1],
            (row_policies, col_policies),
            pw_err_values,
            title=None,
            vrange=None,
            valfmt=valfmt
        )
        fig.tight_layout()


def plot_pairwise_population_comparison(plot_df,
                                        y_key: str,
                                        pop_key: str,
                                        policy_key: str,
                                        coplayer_pop_key: str,
                                        coplayer_policy_key: str,
                                        vrange=None,
                                        figsize=(20, 20),
                                        valfmt=None,
                                        average_duplicates: bool = True,
                                        duplicate_warning: bool = False):
    """Plot results for each policy-seed pairings.

    This produces a grid of (grid)-plots:

    Outer-grid: pop X pop
    Inner-grid: policy X policy

    It is possible that the there are multiple pairings of the same
    (policy, pop) matchup.
    E.g. (agent 0 pi_0 vs agent 1 pi_1) and (agent 0 pi_1 vs agent 1 pi_0).
    In some cases we can just take the average of the two (e.g. when looking
    at times or returns). In such cases use `average_duplicates=True`.
    For cases where we can't take the average (e.g. for getting exp_id), set
    `average_duplicates=False`, in which case the first entry will be used.
    """
    if duplicate_warning:
        if average_duplicates:
            print("Averaging duplicates. FYI")
        else:
            print("Not averaging duplicates. FYI.")

    pop_ids = plot_df[pop_key].unique().tolist()
    pop_ids.sort()
    co_pop_ids = plot_df[coplayer_pop_key].unique().tolist()
    co_pop_ids.sort()

    policies = plot_df[policy_key].unique().tolist()
    policies.sort()
    co_policies = plot_df[coplayer_policy_key].unique().tolist()
    co_policies.sort()

    fig, axs = plt.subplots(
        nrows=len(pop_ids), ncols=len(co_pop_ids), figsize=figsize
    )

    for (row_pop, col_pop) in product(pop_ids, co_pop_ids):
        row_pop_idx = pop_ids.index(row_pop)
        col_pop_idx = co_pop_ids.index(col_pop)

        pw_values = np.zeros((len(policies), len(policies)))
        for (row_policy, col_policy) in product(policies, co_policies):
            row_policy_idx = policies.index(row_policy)
            col_policy_idx = co_policies.index(col_policy)

            y = plot_df[
                (plot_df[policy_key] == row_policy)
                & (plot_df[pop_key] == row_pop)
                & (plot_df[coplayer_policy_key] == col_policy)
                & (plot_df[coplayer_pop_key] == col_pop)
            ][y_key].mean()

            if y is not np.nan and valfmt is None:
                if isinstance(y, float):
                    valfmt = "{x:.2f}"
                if isinstance(y, int):
                    valfmt = "{x}"

            pw_values[row_policy_idx][col_policy_idx] = y

        ax = axs[row_pop_idx][col_pop_idx]
        plot_pairwise_heatmap(
            ax,
            (policies, co_policies),
            pw_values,
            title=None,
            vrange=vrange,
            valfmt=valfmt
        )

        if row_pop_idx == 0:
            ax.set_title(col_pop)
        if col_pop_idx == 0:
            ax.set_ylabel(row_pop)


def get_all_mean_pairwise_values(plot_df,
                                 y_key: str,
                                 policy_key: str,
                                 pop_key: str,
                                 coplayer_policy_key: str,
                                 coplayer_pop_key: str):
    """Get mean pairwise values for all policies."""
    policies = plot_df[policy_key].unique().tolist()
    policies.sort()
    co_policies = plot_df[coplayer_policy_key].unique().tolist()
    co_policies.sort()
    pop_ids = plot_df[pop_key].unique().tolist()
    pop_ids.sort()
    co_pop_ids = plot_df[coplayer_pop_key].unique().tolist()
    co_pop_ids.sort()

    xp_pw_returns = np.zeros((len(policies), len(co_policies)))
    sp_pw_returns = np.zeros((len(policies), len(co_policies)))

    for (row_policy, col_policy) in product(policies, co_policies):
        row_policy_idx = policies.index(row_policy)
        col_policy_idx = co_policies.index(col_policy)

        sp_values = []
        xp_values = []

        for (row_pop, col_pop) in product(pop_ids, co_pop_ids):
            ys = plot_df[
                (plot_df[policy_key] == row_policy)
                & (plot_df[pop_key] == row_pop)
                & (plot_df[coplayer_policy_key] == col_policy)
                & (plot_df[coplayer_pop_key] == col_pop)
            ][y_key]
            y = ys.mean()

            if row_pop == col_pop:
                sp_values.append(y)
            else:
                xp_values.append(y)

        sp_pw_returns[row_policy_idx][col_policy_idx] = np.nanmean(sp_values)
        xp_pw_returns[row_policy_idx][col_policy_idx] = np.nanmean(xp_values)

    return (policies, co_policies), sp_pw_returns, xp_pw_returns


def plot_mean_pairwise_comparison(plot_df,
                                  y_key: str,
                                  policy_key: str,
                                  pop_key: str,
                                  coplayer_policy_key: str,
                                  coplayer_pop_key: str,
                                  vrange: Optional[Tuple[float, float]] = None,
                                  figsize=(12, 6),
                                  valfmt=None):
    """Plot mean pairwise comparison of policies for given y variable."""
    policy_ids, sp_values, xp_values = get_all_mean_pairwise_values(
        plot_df,
        y_key,
        policy_key=policy_key,
        pop_key=pop_key,
        coplayer_policy_key=coplayer_policy_key,
        coplayer_pop_key=coplayer_pop_key
    )

    if vrange is None:
        min_value = np.nanmin([np.nanmin(sp_values), np.nanmin(xp_values)])
        max_value = np.nanmax([np.nanmax(sp_values), np.nanmax(xp_values)])
        vrange = (min_value, max_value)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    plot_pairwise_heatmap(
        axs[0],
        policy_ids,
        sp_values,
        title="Same-Play",
        vrange=vrange,
        valfmt=valfmt
    )
    plot_pairwise_heatmap(
        axs[1],
        policy_ids,
        xp_values,
        title="Cross-Play",
        vrange=vrange,
        valfmt=valfmt
    )

    pw_diff = sp_values - xp_values
    plot_pairwise_heatmap(
        axs[2],
        policy_ids,
        pw_diff,
        title="Difference",
        vrange=(np.nanmin(pw_diff), np.nanmax(pw_diff)),
        valfmt=valfmt
    )

    fig.tight_layout()
    fig.suptitle(y_key)
    return fig

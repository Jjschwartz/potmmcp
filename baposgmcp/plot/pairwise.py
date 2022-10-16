from itertools import permutations, product
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from baposgmcp.plot.heatmap import plot_pairwise_heatmap
from baposgmcp.plot.utils import filter_exps_by, filter_by


def get_pairwise_values(plot_df,
                        y_key: str,
                        policy_key: str = "policy_id",
                        coplayer_policy_key: str = "coplayer_policy_id",
                        policies: Optional[List[str]] = None,
                        coplayer_policies: Optional[List[str]] = None,
                        average_duplicates: bool = True,
                        duplicate_warning: bool = False):
    """Get values for each policy pairing."""
    if policies:
        plot_df = plot_df[plot_df[policy_key].isin(policies)]

    policies = plot_df[policy_key].unique().tolist()
    policies.sort()

    if coplayer_policies is None:
        coplayer_policies = plot_df[coplayer_policy_key].unique().tolist()
        coplayer_policies.sort()

    plot_df = plot_df[plot_df[coplayer_policy_key].isin(coplayer_policies)]
    gb = plot_df.groupby([policy_key, coplayer_policy_key])

    pw_values = np.zeros((len(policies), len(coplayer_policies)))
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
                             policies: Optional[List[str]] = None,
                             coplayer_policies: Optional[List[str]] = None,
                             policy_labels: Optional[Dict[str, str]] = None,
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
        policy_key,
        coplayer_policy_key=coplayer_policy_key,
        policies=policies,
        coplayer_policies=coplayer_policies,
        average_duplicates=average_duplicates,
        duplicate_warning=duplicate_warning
    )

    if policy_labels:
        row_policy_labels = [policy_labels.get(k, k) for k in row_policies]
        col_policy_labels = [policy_labels.get(k, k) for k in col_policies]
    else:
        row_policy_labels = row_policies
        col_policy_labels = col_policies

    plot_pairwise_heatmap(
        axs[0][0],
        (row_policy_labels, col_policy_labels),
        pw_values,
        title=None,
        vrange=vrange,
        valfmt=valfmt
    )

    if y_err_key:
        pw_err_values, _ = get_pairwise_values(
            plot_df,
            y_err_key,
            policy_key,
            coplayer_policy_key=coplayer_policy_key,
            policies=policies,
            coplayer_policies=coplayer_policies,
            average_duplicates=average_duplicates,
            duplicate_warning=duplicate_warning
        )

        plot_pairwise_heatmap(
            axs[0][1],
            (row_policy_labels, col_policy_labels),
            pw_err_values,
            title=None,
            vrange=None,
            valfmt=valfmt
        )
        fig.tight_layout()
    return fig, axs


def plot_pairwise_population_comparison(plot_df,
                                        y_key: str,
                                        pop_key: str,
                                        policy_key: str,
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

    population_ids = plot_df[pop_key].unique().tolist()
    population_ids.sort()

    policies = plot_df[policy_key].unique().tolist()
    policies.sort()

    agent_ids = plot_df["agent_id"].unique()
    agent_ids.sort()

    fig, axs = plt.subplots(
        nrows=len(population_ids), ncols=len(population_ids), figsize=figsize
    )

    for (row_pop, col_pop) in product(population_ids, population_ids):
        row_pop_idx = population_ids.index(row_pop)
        col_pop_idx = population_ids.index(col_pop)

        pw_values = np.zeros((len(policies), len(policies)))
        for (row_policy, col_policy) in product(policies, policies):
            row_policy_idx = policies.index(row_policy)
            col_policy_idx = policies.index(col_policy)

            ys = []
            for (a0, a1) in permutations(agent_ids):
                col_policy_df = filter_exps_by(
                    plot_df,
                    [
                        ("agent_id", "==", a0),
                        (pop_key, "==", col_pop),
                        (policy_key, "==", col_policy)
                    ]
                )
                pairing_df = filter_by(
                    col_policy_df,
                    [
                        ("agent_id", "==", a1),
                        (pop_key, "==", row_pop),
                        (policy_key, "==", row_policy)
                    ]
                )
                pairing_y_vals = pairing_df[y_key].unique()
                pairing_y_vals.sort()

                if len(pairing_y_vals) == 1:
                    ys.append(pairing_y_vals[0])
                elif len(pairing_y_vals) > 1:
                    ys.append(pairing_y_vals[0])
                    if duplicate_warning:
                        print("More than 1 experiment found for pairing:")
                        print(
                            f"({policy_key}={row_policy}, {pop_key}={row_pop},"
                            f" agent_id={a1}) vs ({policy_key}={col_policy}, "
                            f"{pop_key}={col_pop}, agent_id={a0}): "
                            f"{pairing_y_vals}"
                        )
                        print("Plotting only the first value.")

            if len(ys) == 0:
                y = np.nan
            elif len(ys) > 1 and not average_duplicates:
                y = ys[0]
            else:
                y = np.mean(ys)

            if y is not np.nan and valfmt is None:
                if isinstance(y, float):
                    valfmt = "{x:.2f}"
                if isinstance(y, int):
                    valfmt = "{x}"

            pw_values[row_policy_idx][col_policy_idx] = y

        ax = axs[row_pop_idx][col_pop_idx]
        plot_pairwise_heatmap(
            ax,
            (policies, policies),
            pw_values,
            title=None,
            vrange=vrange,
            valfmt=valfmt
        )

        if row_pop_idx == 0:
            ax.set_title(col_pop)
        if col_pop_idx == 0:
            ax.set_ylabel(row_pop)


def get_conditional_pairwise_values(plot_df,
                                    row_conds: List,
                                    row_pop_key: str,
                                    col_conds: List,
                                    col_pop_key: str,
                                    y_key: str
                                    ) -> Tuple[List, List, np.ndarray]:
    """Get pairwise values of y variable.

    Returns a 2D np.array where each cell is the value of y variable for
    a given value pairing of (row_pop_key, col_pop_key) variables.

    Also returns ordered list of row and column labels.

    Additionally, constrains each row and col with additional conditions.
    """
    row_df = filter_by(plot_df, row_conds)
    row_pops = row_df[row_pop_key].unique()
    row_pops.sort()

    col_df = filter_by(plot_df, col_conds)
    col_pops = col_df[col_pop_key].unique()
    col_pops.sort()

    agent_ids = plot_df["agent_id"].unique()
    agent_ids.sort()

    pw_values = np.zeros((len(row_pops), len(col_pops)))

    for c, c_pop in enumerate(col_pops):
        for r, r_pop in enumerate(row_pops):
            ys = []
            for (a0, a1) in permutations(agent_ids):
                c_pop_conds = [
                    ("agent_id", "==", a0), (col_pop_key, "==", c_pop)
                ]
                c_pop_conds = col_conds + c_pop_conds
                c_pop_df = filter_exps_by(plot_df, c_pop_conds)

                r_pop_conds = [
                    ("agent_id", "==", a1), (row_pop_key, "==", r_pop)
                ]
                r_pop_conds = row_conds + r_pop_conds
                r_pop_df = filter_by(c_pop_df, r_pop_conds)

                y = r_pop_df[y_key].tolist()
                if len(y) > 0:
                    ys.extend(y)

            if len(ys) == 0:
                # do this so we can see where error is
                y = np.nan
            else:
                # might have more than a single value since we can have
                # the same (row_val, col_val) pairing for each permutation of
                # agents
                y = np.mean(ys)
            pw_values[r][c] = y

    return (row_pops, col_pops), pw_values


def get_mean_pairwise_population_values(plot_df,
                                        row_conds: List,
                                        row_pop_key: str,
                                        col_conds: List,
                                        col_pop_key: str,
                                        y_key: str) -> Tuple[float, float]:
    """Get pairwise mean values of y variable (y_key) over populations.

    Note this involves taking:
    - for each row_pop - get the average value for each col_pop
    - take average of row_pop averages

    Outputs:
    1. mean for self-play ((row_policy, row_pop) == (col_policy, col_pop)),
       may be np.nan
    2. mean for cross-play
    """
    pops, pw_returns = get_conditional_pairwise_values(
        plot_df,
        row_conds=row_conds,
        row_pop_key=row_pop_key,
        col_conds=col_conds,
        col_pop_key=col_pop_key,
        y_key=y_key
    )

    xp_values = []
    sp_values = []
    for r, row_pop in enumerate(pops[0]):
        for c, col_pop in enumerate(pops[1]):
            v = pw_returns[r][c]
            if np.isnan(v):
                continue
            if row_pop == col_pop:
                sp_values.append(v)
            else:
                xp_values.append(v)

    return np.mean(sp_values), np.mean(xp_values)


def get_all_mean_pairwise_values(plot_df,
                                 y_key: str,
                                 policy_key: str,
                                 pop_key: str):
    """Get mean pairwise values for all policies."""
    policies = plot_df[policy_key].unique().tolist()
    policies.sort()

    xp_pw_returns = np.zeros((len(policies), len(policies)))
    sp_pw_returns = np.zeros((len(policies), len(policies)))

    for r, row_policy_id in enumerate(policies):
        for c, col_policy_id in enumerate(policies):
            sp_return, xp_return = get_mean_pairwise_population_values(
                plot_df,
                row_conds=[(policy_key, "==", row_policy_id)],
                row_pop_key=pop_key,
                col_conds=[(policy_key, "==", col_policy_id)],
                col_pop_key=pop_key,
                y_key=y_key
            )

            sp_pw_returns[r][c] = sp_return
            xp_pw_returns[r][c] = xp_return

    return (policies, policies), sp_pw_returns, xp_pw_returns


def plot_mean_pairwise_comparison(plot_df,
                                  y_key: str,
                                  policy_key: str,
                                  pop_key: str,
                                  vrange: Optional[Tuple[float, float]] = None
                                  ):
    """Plot mean pairwise comparison of policies for given y variable."""
    policy_ids, sp_values, xp_values = get_all_mean_pairwise_values(
        plot_df, y_key, policy_key, pop_key
    )

    if vrange is None:
        min_value = np.nanmin([np.nanmin(sp_values), np.nanmin(xp_values)])
        max_value = np.nanmax([np.nanmax(sp_values), np.nanmax(xp_values)])
        vrange = (min_value, max_value)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    plot_pairwise_heatmap(
        axs[0], policy_ids, sp_values, title="Same-Play", vrange=vrange
    )
    plot_pairwise_heatmap(
        axs[1], policy_ids, xp_values, title="Cross-Play", vrange=vrange
    )

    pw_diff = sp_values - xp_values
    plot_pairwise_heatmap(
        axs[2],
        policy_ids,
        pw_diff,
        title="Difference",
        vrange=(np.nanmin(pw_diff), np.nanmax(pw_diff))
    )

    fig.tight_layout()
    fig.suptitle(y_key)
    return fig

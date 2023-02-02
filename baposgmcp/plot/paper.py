from itertools import product
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


plt.rc("font", size=18)  # controls default text size
plt.rc("axes", titlesize=18)  # fontsize of the title
plt.rc("axes", labelsize=18)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=12)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=12)  # fontsize of the y tick labels
plt.rc("legend", fontsize=18)  # fontsize of the legend
plt.rc("legend", title_fontsize=16)  # fontsize of the legend title


def get_base_plot_kwargs():
    """Get base formatting kwargs for plot."""
    fig_kwargs = {"figsize": (6, 6)}
    base_plot_kwargs = {
        "subplot_kwargs": {
            "xlabel": "Number of Simulations",
        },
        "legend_kwargs": {"fontsize": 14, "title_fontsize": 14},
        "fig_kwargs": fig_kwargs,
    }
    return base_plot_kwargs


def plot_performance_vs_num_sims(
    plot_df: pd.DataFrame,
    ax: Axes,
    y_key: str,
    y_err_key: str,
    policy_key: str,
    policy_prefixes: List[str],
    pi_label_map: Optional[Dict[str, str]] = None,
):
    """Plot expected values for different policies by num_sims.

    Assumes policies with sims have IDs that include
    "_numsims<n>"
    """
    policy_ids = plot_df[policy_key].unique().tolist()
    policy_ids.sort()

    if pi_label_map is None:
        pi_label_map = {}

    values_by_pi = {}
    all_num_sims = set()
    for prefix in policy_prefixes:
        values_by_pi[prefix] = {"y": {}, "y_err": {}}
        for i, policy_id in enumerate(policy_ids):
            if not policy_id.startswith(prefix):
                continue

            tokens = policy_id.split("_")
            num_sims = None
            for t in tokens:
                if t.startswith("numsims"):
                    num_sims = int(t.replace("numsims", ""))

            if num_sims is not None:
                all_num_sims.add(num_sims)

            pi_df = plot_df[plot_df[policy_key] == policy_id]
            value = pi_df[y_key].values[0]
            err_value = pi_df[y_err_key].values[0]
            values_by_pi[prefix]["y"][num_sims] = value
            values_by_pi[prefix]["y_err"][num_sims] = err_value

    all_num_sims = list(all_num_sims)
    all_num_sims.sort()

    for prefix in policy_prefixes:
        y_map = values_by_pi[prefix]["y"]
        y_err_map = values_by_pi[prefix]["y_err"]
        num_sims = list(y_map)
        num_sims.sort()

        if len(num_sims) == 1:
            # non-sim policy
            y = np.full(len(all_num_sims), y_map[None])
            y_err = np.full(len(all_num_sims), y_err_map[None])
            num_sims = all_num_sims
        else:
            y = np.array([y_map[n] for n in num_sims])
            y_err = np.array([y_err_map[n] for n in num_sims])

        label = pi_label_map.get(prefix, prefix)

        ax.plot(num_sims, y, label=label)
        ax.fill_between(num_sims, y - y_err, y + y_err, alpha=0.2)


def plot_multiple_performance_vs_num_sims(
    plot_df: pd.DataFrame,
    y_key: str,
    y_err_key: str,
    policy_prefixes,
    pi_label_map=None,
    subplot_kwargs=None,
    legend_kwargs=None,
    fig_kwargs=None,
) -> Tuple[Figure, List[List[Axes]]]:
    """Create multiple performance vs num sims plots."""
    if not isinstance(policy_prefixes[0], list):
        policy_prefixes = [policy_prefixes]

    if not subplot_kwargs:
        subplot_kwargs = {}

    if not legend_kwargs:
        legend_kwargs = {}

    num_rows = len(policy_prefixes)
    num_cols = 1

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        squeeze=False,
        subplot_kw=subplot_kwargs,
        **fig_kwargs,
    )

    for row_axs, prefix_list in zip(axs, policy_prefixes):
        ax = row_axs[0]
        plot_performance_vs_num_sims(
            plot_df,
            ax,
            y_key=y_key,
            y_err_key=y_err_key,
            policy_key="policy_id",
            policy_prefixes=prefix_list,
            pi_label_map=pi_label_map,
        )
        ax.legend(**legend_kwargs)

    return fig, axs


def plot_meta_policy_vs_num_sims(
    plot_df: pd.DataFrame,
    ax: Axes,
    y_key: str,
    y_err_key: str,
    alg_id_key: str = "alg_id",
    meta_pi_label_map=None,
):
    """Plot expected values for different meta-policies by num_sims."""
    assert len(plot_df[alg_id_key].unique()) == 1
    if not meta_pi_label_map:
        meta_pi_label_map = {}

    all_meta_pis = plot_df["meta_pi"].unique().tolist()
    all_meta_pis.sort()

    for meta_pi in all_meta_pis:
        a_df = plot_df[plot_df["meta_pi"] == meta_pi]
        a_df = a_df.sort_values(by="num_sims")

        x = a_df["num_sims"]
        y = a_df[y_key]
        y_err = a_df[y_err_key]
        label = meta_pi_label_map.get(meta_pi, meta_pi)

        ax.plot(x, y, label=label)
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)


def plot_meta_policy_for_nonsim(
    plot_df: pd.DataFrame,
    ax: Axes,
    y_key: str,
    y_err_key: str,
    alg_id_key: str = "alg_id",
    meta_pi_label_map=None,
):
    """Plot expected values vs meta-policies for non-sim policies."""
    assert len(plot_df[alg_id_key].unique()) == 1
    if not meta_pi_label_map:
        meta_pi_label_map = {}

    all_meta_pis = plot_df["meta_pi"].unique().tolist()
    all_meta_pis.sort()

    xs = np.arange(len(all_meta_pis))
    ys = plot_df[y_key]
    y_errs = plot_df[y_err_key]
    labels = [meta_pi_label_map.get(p, p) for p in all_meta_pis]

    ax.bar(xs, ys, yerr=y_errs, tick_label=labels)


def plot_multiple_meta_policy_vs_num_sims(
    plot_df: pd.DataFrame,
    y_key: str,
    y_err_key: str,
    alg_id_key: str = "alg_id",
    meta_pi_label_map=None,
    subplot_kwargs=None,
    legend_kwargs=None,
    fig_kwargs=None,
    set_title: bool = False,
) -> Tuple[Figure, List[List[Axes]]]:
    """Create multiple meta-policy vs num sims plots."""
    alg_ids = plot_df[alg_id_key].unique().tolist()
    alg_ids.sort()

    if not subplot_kwargs:
        subplot_kwargs = {}

    if not legend_kwargs:
        legend_kwargs = {}

    num_rows = len(alg_ids)
    num_cols = 1

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        squeeze=False,
        subplot_kw=subplot_kwargs,
        **fig_kwargs,
    )

    for row_axs, alg_id in zip(axs, alg_ids):
        ax = row_axs[0]
        alg_df = plot_df[plot_df[alg_id_key] == alg_id]

        if len(alg_df["num_sims"].unique()) == 1:
            plot_meta_policy_for_nonsim(
                alg_df,
                ax,
                y_key=y_key,
                y_err_key=y_err_key,
                alg_id_key=alg_id_key,
                meta_pi_label_map=meta_pi_label_map,
            )
        else:
            plot_meta_policy_vs_num_sims(
                alg_df,
                ax,
                y_key=y_key,
                y_err_key=y_err_key,
                alg_id_key=alg_id_key,
                meta_pi_label_map=meta_pi_label_map,
            )
            ax.legend(**legend_kwargs)

        if set_title:
            ax.set_title(alg_id)

    return fig, axs


def plot_truncated_vs_num_sims_by_alg(
    plot_df: pd.DataFrame,
    ax: Axes,
    y_key: str,
    y_err_key: str,
    truncated_label_map=None,
    alg_id_key: str = "alg_id",
):
    """Plot expected values for truncated vs untruncated by num_sims."""
    assert len(plot_df[alg_id_key].unique()) == 1

    if not truncated_label_map:
        truncated_label_map = {}

    all_trunc = plot_df["truncated"].unique().tolist()
    all_trunc.sort()

    for trunc in all_trunc:
        a_df = plot_df[plot_df["truncated"] == trunc]
        a_df = a_df.sort_values(by="num_sims")
        x = a_df["num_sims"]
        y = a_df[y_key]
        y_err = a_df[y_err_key]

        ax.plot(x, y, label=truncated_label_map.get(trunc, trunc))
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)


def plot_multiple_truncated_vs_num_sims_by_alg(
    plot_df: pd.DataFrame,
    y_key: str,
    y_err_key: str,
    alg_id_key: str = "alg_id",
    truncated_label_map=None,
    subplot_kwargs=None,
    legend_kwargs=None,
    fig_kwargs=None,
    set_title: bool = False,
) -> Tuple[Figure, List[List[Axes]]]:
    """Create multiple truncated vs num sims plots."""
    alg_ids = plot_df[alg_id_key].unique()
    alg_ids.sort()

    num_rows = len(alg_ids)
    num_cols = 1

    if not subplot_kwargs:
        subplot_kwargs = {}

    if not legend_kwargs:
        legend_kwargs = {}

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        squeeze=False,
        subplot_kw=subplot_kwargs,
        **fig_kwargs,
    )

    for row_axs, alg_id in zip(axs, alg_ids):
        ax = row_axs[0]
        alg_df = plot_df[plot_df[alg_id_key] == alg_id]

        plot_truncated_vs_num_sims_by_alg(
            alg_df,
            ax,
            y_key=y_key,
            y_err_key=y_err_key,
            alg_id_key=alg_id_key,
            truncated_label_map=truncated_label_map,
        )
        ax.legend(**legend_kwargs)
        if set_title:
            ax.set_title(alg_id)

    return fig, axs


def plot_action_selection_vs_num_sims(
    plot_df: pd.DataFrame,
    ax: Axes,
    y_key: str,
    y_err_key: str,
    alg_id_key="alg_id",
    act_sel_key="action_selection",
    label_map=None,
    subplot_kwargs=None,
    legend_kwargs=None,
    fig_kwargs=None,
    set_title: bool = False,
):
    """Plot expected values for action selction strategies by num_sims."""
    if not subplot_kwargs:
        subplot_kwargs = {}

    if not legend_kwargs:
        legend_kwargs = {}

    fig, axs = plt.subplots(
        nrows=1, ncols=1, squeeze=False, subplot_kw=subplot_kwargs, **fig_kwargs
    )

    if not label_map:
        label_map = {}

    alg_ids = plot_df[alg_id_key].unique().tolist()
    alg_ids.sort()

    all_act_sel = plot_df[act_sel_key].unique().tolist()
    all_act_sel.sort()

    for alg_id, act_sel in product(alg_ids, all_act_sel):
        a_df = plot_df[
            (plot_df[act_sel_key] == act_sel) & (plot_df[alg_id_key] == alg_id)
        ]
        if len(a_df) == 0:
            continue

        a_df = a_df.sort_values(by="num_sims")
        x = a_df["num_sims"]
        y = a_df[y_key]
        y_err = a_df[y_err_key]

        ax.plot(x, y, label=label_map.get((alg_id, act_sel), f"{alg_id} - {act_sel}"))
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)


def plot_expected_belief_stat_by_step(
    plot_df: pd.DataFrame,
    ax: Axes,
    y_key_prefix: str,
    step_limit: int,
    other_agent_id: int,
    y_suffix: str = "mean",
    y_err_suffix: str = "CI",
):
    """Plot expected value of a belief stat w.r.t policy prior by ep step."""
    xs = np.arange(0, step_limit)

    y_keys = [f"{y_key_prefix}_{other_agent_id}_{t}" for t in range(step_limit)]
    num_sims = plot_df["num_sims"].unique().tolist()
    num_sims.sort()

    for n in num_sims:
        n_df = plot_df[plot_df["num_sims"] == n]
        y = np.zeros(step_limit)
        y_err = np.zeros(step_limit)

        for t in range(step_limit):
            y_t = n_df[f"{y_keys[t]}_{y_suffix}"]
            y_err_t = n_df[f"{y_keys[t]}_{y_err_suffix}"]
            if len(y_t):
                assert len(y_t) == 1, f"{y_t}"
                y[t] = y_t.values[0]
                y_err[t] = y_err_t.values[0]

        ax.plot(xs, y, label=n)
        ax.fill_between(xs, y - y_err, y + y_err, alpha=0.2)


def plot_multiple_belief_stats(
    plot_df: pd.DataFrame,
    y_key_prefix: str,
    step_limit: int,
    other_agent_id: int,
    y_suffix: str = "mean",
    y_err_suffix: str = "CI",
    alg_id_key: str = "alg_id",
    subplot_kwargs=None,
    legend_kwargs=None,
    fig_kwargs=None,
    set_title: bool = False,
) -> Tuple[Figure, List[List[Axes]]]:
    """Create multiple belief stat vs step by num sims plots."""
    alg_ids = plot_df[alg_id_key].unique().tolist()
    alg_ids.sort()

    if not subplot_kwargs:
        subplot_kwargs = {}

    if not legend_kwargs:
        legend_kwargs = {}

    num_rows = len(alg_ids)
    num_cols = 1

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        squeeze=False,
        subplot_kw=subplot_kwargs,
        **fig_kwargs,
    )

    for row_axs, alg_id in zip(axs, alg_ids):
        ax = row_axs[0]
        alg_df = plot_df[plot_df[alg_id_key] == alg_id]
        plot_expected_belief_stat_by_step(
            alg_df,
            ax,
            y_key_prefix=y_key_prefix,
            step_limit=step_limit,
            other_agent_id=other_agent_id,
            y_suffix=y_suffix,
            y_err_suffix=y_err_suffix,
        )
        ax.legend(**legend_kwargs)

        if set_title:
            ax.set_title(alg_id)

    return fig, axs


def plot_performance_vs_search_time(
    plot_df: pd.DataFrame,
    ax: Axes,
    y_key: str,
    y_err_key: str,
    policy_key: str,
    policy_prefixes: List[str],
    pi_label_map: Optional[Dict[str, str]] = None,
):
    """Plot expected values for different policies by mean search time."""
    policy_ids = plot_df[policy_key].unique().tolist()
    policy_ids.sort()

    if pi_label_map is None:
        pi_label_map = {}

    values_by_pi = {}
    min_time, max_time = float("inf"), -1.0
    for prefix in policy_prefixes:
        values_by_pi[prefix] = {"y": [], "y_err": []}
        for i, policy_id in enumerate(policy_ids):
            if not policy_id.startswith(prefix):
                continue

            pi_df = plot_df[plot_df[policy_key] == policy_id]
            search_time = pi_df["search_time_mean"].values[0]

            if search_time < min_time:
                min_time = search_time
            if search_time > max_time:
                max_time = search_time

            value = pi_df[y_key].values[0]
            err_value = pi_df[y_err_key].values[0]
            values_by_pi[prefix]["y"].append((search_time, value))
            values_by_pi[prefix]["y_err"].append((search_time, err_value))

    xs = np.linspace(min_time, max_time, num=10)

    for prefix in policy_prefixes:
        y_list = values_by_pi[prefix]["y"]
        y_err_list = values_by_pi[prefix]["y_err"]
        y_list.sort()
        y_err_list.sort()

        if len(y_list) == 1:
            # non-sim policy
            y = np.full(len(xs), y_list[0][1])
            y_times = xs
            y_err = np.full(len(xs), y_err_list[0][1])
            y_err_times = xs
        else:
            y = np.array([v[1] for v in y_list])
            y_times = np.array([v[0] for v in y_list])
            y_err = np.array([v[1] for v in y_err_list])
            y_err_times = np.array([v[0] for v in y_err_list])

        label = pi_label_map.get(prefix, prefix)

        ax.plot(y_times, y, label=label)
        ax.fill_between(y_err_times, y - y_err, y + y_err, alpha=0.2)


def plot_multiple_performance_vs_search_time(
    plot_df: pd.DataFrame,
    y_key: str,
    y_err_key: str,
    policy_prefixes,
    pi_label_map=None,
    subplot_kwargs=None,
    legend_kwargs=None,
    fig_kwargs=None,
):
    """Create multiple performance vs num sims plots."""
    if not isinstance(policy_prefixes[0], list):
        policy_prefixes = [policy_prefixes]

    if not subplot_kwargs:
        subplot_kwargs = {}

    if not legend_kwargs:
        legend_kwargs = {}

    num_rows = len(policy_prefixes)
    num_cols = 1

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        squeeze=False,
        subplot_kw=subplot_kwargs,
        **fig_kwargs,
    )

    for row_axs, prefix_list in zip(axs, policy_prefixes):
        ax = row_axs[0]
        plot_performance_vs_search_time(
            plot_df,
            ax,
            y_key=y_key,
            y_err_key=y_err_key,
            policy_key="policy_id",
            policy_prefixes=prefix_list,
            pi_label_map=pi_label_map,
        )
        ax.legend(**legend_kwargs)

    return fig, axs

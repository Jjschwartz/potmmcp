from itertools import product
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


algname = "POTMMCP"
baselinealgname = "I-POMCP-PF"

# For UAI paper formatting
PAGE_WIDTH = 6.75  # inches
PAGE_COL_WIDTH = (6.75 - 0.25) / 2

TINY_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_base_plot_kwargs():
    """Get base formatting kwargs for plot."""
    base_plot_kwargs = {
        "subplot_kwargs": {
            "xlabel": "Number of Simulations",
        },
        "legend_kwargs": {},
        "fig_kwargs": {"figsize": (6, 6)},
    }
    return base_plot_kwargs


def plot_performance(
    plot_df: pd.DataFrame,
    ax: Axes,
    x_key: str,
    y_key: str,
    y_err_key: str,
    policy_key: str,
    policy_prefixes: List[str],
    pi_label_map: Optional[Dict[str, str]] = None,
    constant_policy_prefixes: Optional[List[str]] = None,
):
    """Plot expected values for different policies by x key."""
    policy_ids = plot_df[policy_key].unique().tolist()
    policy_ids.sort()

    if pi_label_map is None:
        pi_label_map = {}

    if constant_policy_prefixes is None:
        constant_policy_prefixes = []

    values_by_pi = {}
    x_values = set()
    for prefix in policy_prefixes:
        values_by_pi[prefix] = {"y": [], "y_err": []}
        for i, policy_id in enumerate(policy_ids):
            if not policy_id.startswith(prefix):
                continue

            pi_df = plot_df[plot_df[policy_key] == policy_id]

            if any(policy_id.startswith(p) for p in constant_policy_prefixes):
                x = None
            else:
                x = pi_df[x_key].values[0]
                x_values.add(x)

            values_by_pi[prefix]["y"].append((x, pi_df[y_key].values[0]))
            values_by_pi[prefix]["y_err"].append((x, pi_df[y_err_key].values[0]))

    all_x_values = list(x_values)
    all_x_values.sort()

    for prefix in policy_prefixes:
        y_list = values_by_pi[prefix]["y"]
        y_err_list = values_by_pi[prefix]["y_err"]
        y_list.sort()
        y_err_list.sort()

        if len(y_list) == 1:
            # non-sim policy
            y = np.full(len(all_x_values), y_list[0][1])
            x = all_x_values
            y_err = np.full(len(all_x_values), y_err_list[0][1])
        else:
            y = np.array([v[1] for v in y_list])
            x = np.array([v[0] for v in y_list])
            y_err = np.array([v[1] for v in y_err_list])

        label = pi_label_map.get(prefix, prefix)

        ax.plot(x, y, label=label)
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)


def plot_multiple_performance(
    plot_df: pd.DataFrame,
    x_key: str,
    y_key: str,
    y_err_key: str,
    policy_prefixes: List[str],
    pi_label_map: Optional[Dict[str, str]] = None,
    constant_policy_prefixes: Optional[List[str]] = None,
    policy_key: str = "policy_id",
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
        plot_performance(
            plot_df,
            ax,
            x_key=x_key,
            y_key=y_key,
            y_err_key=y_err_key,
            policy_key=policy_key,
            policy_prefixes=prefix_list,
            pi_label_map=pi_label_map,
            constant_policy_prefixes=constant_policy_prefixes
        )
        ax.legend(**legend_kwargs)

    return fig, axs


def plot_meta_policy_performance(
    plot_df: pd.DataFrame,
    ax: Axes,
    x_key: str,
    y_key: str,
    y_err_key: str,
    alg_id_key: str = "alg_id",
    meta_pi_label_map: Optional[Dict[str, str]] = None,
):
    """Plot expected values for different meta-policies by num_sims."""
    assert len(plot_df[alg_id_key].unique()) == 1
    if not meta_pi_label_map:
        meta_pi_label_map = {}

    all_meta_pis = plot_df["meta_pi"].unique().tolist()
    all_meta_pis.sort()

    for meta_pi in all_meta_pis:
        a_df = plot_df[plot_df["meta_pi"] == meta_pi]
        a_df = a_df.sort_values(by=x_key)

        x = a_df[x_key]
        y = a_df[y_key]
        y_err = a_df[y_err_key]
        label = meta_pi_label_map.get(meta_pi, meta_pi)

        ax.plot(x, y, label=label)
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)


def bar_plot_meta_policy_performance(
    plot_df: pd.DataFrame,
    ax: Axes,
    y_key: str,
    y_err_key: str,
    alg_id_key: str = "alg_id",
    meta_pi_label_map: Optional[Dict[str, str]] = None,
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


def plot_multiple_meta_policy_performance(
    plot_df: pd.DataFrame,
    x_key: str,
    y_key: str,
    y_err_key: str,
    alg_id_key: str = "alg_id",
    meta_pi_label_map: Optional[Dict[str, str]] = None,
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

        if len(alg_df[x_key].unique()) == 1:
            bar_plot_meta_policy_performance(
                alg_df,
                ax,
                y_key=y_key,
                y_err_key=y_err_key,
                alg_id_key=alg_id_key,
                meta_pi_label_map=meta_pi_label_map,
            )
        else:
            plot_meta_policy_performance(
                alg_df,
                ax,
                x_key=x_key,
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
    z_key: str,
    y_key_prefix: str,
    step_limit: int,
    other_agent_id: int,
    y_suffix: str = "mean",
    y_err_suffix: str = "CI",
):
    """Plot expected value of a belief stat w.r.t policy prior by ep step."""
    xs = np.arange(0, step_limit)

    y_keys = [f"{y_key_prefix}_{other_agent_id}_{t}" for t in range(step_limit)]
    zs = plot_df[z_key].unique().tolist()
    zs.sort()

    for z in zs:
        z_df = plot_df[plot_df[z_key] == z]
        y = np.zeros(step_limit)
        y_err = np.zeros(step_limit)

        for t in range(step_limit):
            y_t = z_df[f"{y_keys[t]}_{y_suffix}"]
            y_err_t = z_df[f"{y_keys[t]}_{y_err_suffix}"]
            if len(y_t):
                assert len(y_t) == 1, f"{y_t}"
                y[t] = y_t.values[0]
                y_err[t] = y_err_t.values[0]

        ax.plot(xs, y, label=z)
        ax.fill_between(xs, y - y_err, y + y_err, alpha=0.2)


def plot_multiple_belief_stats(
    plot_df: pd.DataFrame,
    z_key: str,
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
    """Create multiple belief stat vs step by z_key plots."""
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
            z_key=z_key,
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

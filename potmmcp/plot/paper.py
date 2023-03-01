"""Functions for generating paper plots."""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


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

    values_by_pi = {}  # type: ignore
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


def plot_meta_policy_performance(
    plot_df: pd.DataFrame,
    ax: Axes,
    x_key: str,
    y_key: str,
    y_err_key: str,
    meta_pi_label_map: Optional[Dict[str, str]] = None,
):
    """Plot expected values for different meta-policies."""
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
    meta_pi_label_map: Optional[Dict[str, str]] = None,
):
    """Plot expected values vs meta-policies for non-sim policies."""
    if not meta_pi_label_map:
        meta_pi_label_map = {}

    all_meta_pis = plot_df["meta_pi"].unique().tolist()
    all_meta_pis.sort()

    xs = np.arange(len(all_meta_pis))
    ys = plot_df[y_key]
    y_errs = plot_df[y_err_key]
    labels = [meta_pi_label_map.get(p, p) for p in all_meta_pis]

    ax.bar(xs, ys, yerr=y_errs, tick_label=labels)


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

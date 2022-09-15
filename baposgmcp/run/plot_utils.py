from typing import List, Optional, Tuple
from itertools import permutations, product

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import posggym


def add_95CI(df: pd.DataFrame) -> pd.DataFrame:
    """Add 95% CI columns to dataframe."""

    def conf_int(row, prefix):
        std = row[f"{prefix}_std"]
        n = row["num_episodes"]
        return 1.96 * (std / np.sqrt(n))

    prefix = ""
    for col in df.columns:
        if not col.endswith("_std"):
            continue
        prefix = col.replace("_std", "")
        df[f"{prefix}_CI"] = df.apply(
            lambda row: conf_int(row, prefix), axis=1
        )
    return df


def add_outcome_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Add proportion columns to dataframe."""

    def prop(row, col_name):
        n = row["num_episodes"]
        total = row[col_name]
        return total / n

    columns = [
        'num_outcome_LOSS',
        'num_outcome_DRAW',
        'num_outcome_WIN',
        'num_outcome_NA'
    ]
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


def import_results(result_file: str,
                   columns_to_drop: Optional[List[str]] = None,
                   clean_policy_id: bool = True
                   ) -> pd.DataFrame:
    """Import experiment results.

    If `clean_policy_id` is True then the environment name will be stripped
    from any policy ID.
    """
    df = pd.read_csv(result_file)

    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1, errors='ignore')

    df = add_95CI(df)
    df = add_outcome_proportions(df)

    if clean_policy_id:
        df = clean_df_policy_ids(df)

    return df


def _sort_and_display(df: pd.DataFrame, key: str, display_name: str):
    values = df[key].unique()
    try:
        values.sort()
    except TypeError:
        pass
    print(f"{display_name}: {values}")


def filter_by(df: pd.DataFrame,
              conds: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Filter dataframe by given conditions.

    Removes any rows that do not meet the conditions.
    """
    query_strs = []
    for (k, op, v) in conds:
        if isinstance(v, str):
            q = f"({k} {op} '{v}')"
        else:
            q = f"({k} {op} {v})"
        query_strs.append(q)
    query = " & ".join(query_strs)

    filtered_df = df.query(query)
    return filtered_df


def filter_exps_by(df: pd.DataFrame,
                   conds: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Filter experiments in dataframe by given conditions.

    Ensures all rows for an experiment where any row of that experiment
    meets the conditions are in the resulting dataframe.

    Removes any rows that do not meet the conditions and are not part of
    an experiment where at least one row (i.e. agent) meets the condition.
    """
    filtered_df = filter_by(df, conds)
    exp_ids = filtered_df["exp_id"].unique()
    df = df[df["exp_id"].isin(exp_ids)]
    return df


def plot_environment(env_name: str):
    """Display rendering of the environment."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    # Turn off x/y axis numbering/ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    env = posggym.make(env_name)
    env_img, _ = env.render(mode='rgb_array')

    imshow_obj = ax.imshow(
        env_img, interpolation='bilinear', origin='upper'
    )
    imshow_obj.set_data(env_img)

    ax.set_title(env_name)


def heatmap(data,
            row_labels,
            col_labels,
            ax=None,
            show_cbar=True,
            cbar_kw={},
            cbarlabel="",
            **kwargs):
    """Create a heatmap from a numpy array and two lists of labels.

    ref:
    matplotlib.org/stable/gallery/images_contours_and_fields/
    image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    show_cbar
        If true a color bard is displayed, otherwise no colorbar is shown.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(
        top=True, bottom=False, labeltop=True, labelbottom=False
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor"
    )

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im,
                     data=None,
                     valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None,
                     **textkw):
    """Annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.

    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_pairwise_heatmap(ax,
                          labels: Tuple[List[str], List[str]],
                          values: np.ndarray,
                          title: Optional[str] = None,
                          vrange: Optional[Tuple[float, float]] = None,
                          valfmt: Optional[str] = None):
    """Plot pairwise values as a heatmap."""
    # Note numpy arrays by default have (0, 0) in the top-left corner.
    # While matplotlib images are displayed with (0, 0) being the bottom-left
    # corner.
    # To get images looking correct we need to reverse the rows of the array
    # And also the row labels
    values = values[-1::-1]

    if vrange is None:
        vrange = (np.nanmin(values), np.nanmax(values))

    if valfmt is None:
        valfmt = "{x:.2f}"

    im, cbar = heatmap(
        data=values,
        row_labels=reversed(labels[0]),
        col_labels=labels[1],
        ax=ax,
        show_cbar=False,
        cmap="viridis",
        vmin=vrange[0],
        vmax=vrange[1]
    )

    annotate_heatmap(
        im,
        valfmt=valfmt,
        textcolors=("white", "black"),
        threshold=vrange[0]+(0.2*(vrange[1] - vrange[0]))
    )

    if title:
        ax.set_title(title)


def get_pairwise_values(plot_df,
                        y_key: str,
                        policy_key: str,
                        average_duplicates: bool = True,
                        duplicate_warning: bool = False):
    """Get values for each policy pairing."""
    policies = plot_df[policy_key].unique().tolist()
    policies.sort()

    agent_ids = plot_df["agent_id"].unique()
    agent_ids.sort()

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
                    (policy_key, "==", col_policy)
                ]
            )
            pairing_df = filter_by(
                col_policy_df,
                [
                    ("agent_id", "==", a1),
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
                        f"({policy_key}={row_policy}, agent_id={a1}) vs "
                        f"({policy_key}={col_policy}, agent_id={a0}): "
                        f"{pairing_y_vals}"
                    )
                    print("Plotting only the first value.")

        if len(ys) == 0:
            y = np.nan
        elif len(ys) > 1 and not average_duplicates:
            y = ys[0]
        else:
            y = np.mean(ys)
        pw_values[row_policy_idx][col_policy_idx] = y

    return pw_values, policies


def plot_pairwise_comparison(plot_df,
                             y_key: str,
                             policy_key: str,
                             y_err_key: Optional[str] = None,
                             vrange=None,
                             figsize=(20, 20),
                             valfmt=None,
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
        nrows=1, ncols=ncols, figsize=figsize, squeeze=False
    )

    pw_values, policies = get_pairwise_values(
        plot_df,
        y_key,
        policy_key,
        average_duplicates=average_duplicates,
        duplicate_warning=duplicate_warning
    )

    plot_pairwise_heatmap(
        axs[0][0],
        (policies, policies),
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
            average_duplicates=average_duplicates,
            duplicate_warning=duplicate_warning
        )

        plot_pairwise_heatmap(
            axs[0][1],
            (policies, policies),
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

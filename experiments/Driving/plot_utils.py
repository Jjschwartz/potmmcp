import os.path as osp
from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import posggym

import exp_utils


BAPOSGMCP_HYPERPARAMETERS = [
    "other_policy_prior",
    "truncated",
    "c_init",
    "c_base",
    "reinvigorator",
    "extra_particles_prop",
    "step_limit",
    "epsilon",
]


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


def add_proportions(df: pd.DataFrame,
                    columns: Optional[List[str]] = None,
                    new_column_names: Optional[List[str]] = None
                    ) -> pd.DataFrame:
    """Add proportion columns to dataframe."""
    assert (
        (columns is None and new_column_names is None)
        or (columns is not None and new_column_names is not None)
    )

    def prop(row, col_name):
        n = row["num_episodes"]
        total = row[col_name]
        return total / n

    if columns is None:
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


def _rename_policy_dir(row):
    policy_dir = row["policy_dir"]
    policy_dir = osp.basename(osp.normpath(policy_dir))
    return policy_dir


def _get_k(row):
    pi_name = row["policy_name"]
    if "Random" in pi_name:
        return str(-1)
    if "BAPOSGMCP" in pi_name:
        return "BA"
    try:
        return pi_name.split("_")[-1]
    except Exception:
        raise Exception("Policy name error")


def _get_rollout_k(row):
    pi_name = row["rollout_id"]
    if "Random" in pi_name:
        return str(-1)
    if "BAPOSGMCP" in pi_name:
        return "BA"
    if pi_name == 'None':
        return 'None'
    try:
        return pi_name.split("_")[-1]
    except Exception:
        raise Exception("Policy name error")


def parse_policy_directory(policy_dir: str):
    """Return (alg, env_name, seed, alg_kwargs)."""
    if policy_dir in (None, 'None', np.nan):
        # random or some other fixed policy
        return 'None', 'None', 'None', {}

    # get only final directory in path
    policy_dir = osp.basename(osp.normpath(policy_dir))

    # expect 'train_<alg>_<env_name>_[algkwargs_]seed<int>_<timestamp>'
    tokens = policy_dir.split("_")
    alg = tokens[1].lower()
    env_name = tokens[2]

    seed = 'None'
    for t in tokens:
        if "seed=" in t:
            # for backward compatibility with old naming style
            seed = int(t.split("=")[1])
            break
        elif "seed" in t:
            seed = int(t.replace("seed", ""))
            break

    if alg == "klr":
        # expect 'train_<alg>_<env_name>_k=<k>_[seed=int]_<timestamp>'
        if "k=" in tokens[3]:
            k = int(tokens[3].split("=")[1])
        elif "k" in tokens[3]:
            k = int(tokens[3].replace("k", ""))
        else:
            raise AssertionError(f"Bad policy dir format: {policy_dir}")
        alg_kwargs = {"k": k}
    else:
        alg_kwargs = {}

    return alg, env_name, seed, alg_kwargs


def _get_train_properties(row):
    if "policy_dir" not in row:
        return 'None', 'None', 'None', {}
    return parse_policy_directory(row["policy_dir"])


def _get_train_alg(row):
    return _get_train_properties(row)[0]


def _get_train_env(row):
    return _get_train_properties(row)[1]


def _get_train_seed(row):
    return _get_train_properties(row)[2]


def get_rollout_policy_properties(row):
    """Return (policy_ID, alg, env_name, seed, alg_kwargs)."""
    if row["K"] != "BA":
        return 'None', 'None', 'None', 'None', {}

    alg, env_name, seed, alg_kwargs = parse_policy_directory(
        row["rollout_policy_dir"]
    )
    if alg == "klr":
        rollout_policy_id = "pi_BR"
    elif alg == "sp":
        rollout_policy_id = "pi_SP"
    else:
        rollout_policy_id = "pi_-1"       # random

    return rollout_policy_id, alg, env_name, seed, alg_kwargs


def get_other_policies_properties(row):
    """Return (alg, env_name, seed, alg_kwargs)."""
    if row["K"] != "BA":
        return 'None', 'None', 'None', {}
    return parse_policy_directory(row["other_agent_policy_dir"])


def _get_rollout_policy_id(row):
    return get_rollout_policy_properties(row)[0]


def _get_rollout_policy_alg(row):
    return get_rollout_policy_properties(row)[1]


def _get_rollout_policy_seed(row):
    return get_rollout_policy_properties(row)[3]


def _get_other_agent_alg(row):
    return get_other_policies_properties(row)[0]


def _get_other_agent_seed(row):
    return get_other_policies_properties(row)[2]


def get_coplayer_property(row, plot_df, key):
    """Get properties of coplayer."""
    co_player_df = plot_df[
        (plot_df["exp_id"] == row["exp_id"])
        & (plot_df["agent_id"] != row["agent_id"])
    ]
    return co_player_df[key].unique()[0]



def import_results(result_dir: str,
                   columns_to_drop: List[str],
                   is_baposgmcp_result: bool) -> pd.DataFrame:
    """Import driving experiment results and do some cleaning."""
    result_file = osp.join(
        exp_utils.EXP_RESULTS_DIR, result_dir, "compiled_results.csv"
    )
    df = pd.read_csv(result_file)

    df = df.drop(columns_to_drop, axis=1, errors='ignore')
    df = add_95CI(df)
    df = add_proportions(df)

    df["policy_dir"] = df.apply(_rename_policy_dir, axis=1)
    df["K"] = df.apply(_get_k, axis=1)
    df["train_env_name"] = df.apply(_get_train_env, axis=1)
    df["train_seed"] = df.apply(_get_train_seed, axis=1)
    df["train_alg"] = df.apply(_get_train_alg, axis=1)
    df["coplayer_K"] = df.apply(
        lambda row: get_coplayer_property(row, df, "K"), axis=1
    )
    df["coplayer_train_seed"] = df.apply(
        lambda row: get_coplayer_property(row, df, "train_seed"), axis=1
    )

    if is_baposgmcp_result:
        df["rollout_id"] = df.apply(_get_rollout_policy_id, axis=1)
        df["rollout_K"] = df.apply(_get_rollout_k, axis=1)
        df["rollout_seed"] = df.apply(_get_rollout_policy_seed, axis=1)
        df["rollout_alg"] = df.apply(_get_rollout_policy_alg, axis=1)
        df["other_alg"] = df.apply(_get_other_agent_alg, axis=1)
        df["other_seed"] = df.apply(_get_other_agent_seed, axis=1)
        df["coplayer_num_sims"] = df.apply(
            lambda row: get_coplayer_property(row, df, "num_sims"), axis=1
        )
        df["coplayer_other_seed"] = df.apply(
            lambda row: get_coplayer_property(row, df, "other_seed"), axis=1
        )

    return df


def _sort_and_display(df: pd.DataFrame, key: str, display_name: str):
    values = df[key].unique()
    try:
        values.sort()
    except TypeError:
        pass
    print(f"{display_name}: {values}")


def validate_and_display(df: pd.DataFrame, is_baposgmcp_result: bool):
    """Validate dataframe and display summary."""
    agent_ids = df["agent_id"].unique()
    agent_ids.sort()
    assert len(agent_ids) == 2
    print("Agent IDs:", agent_ids)

    test_envs = df["env_name"].unique()
    assert len(test_envs) == 1
    test_env = test_envs[0]
    print("Test Env:", test_env)

    num_entries = len(df)
    num_exps = len(df["exp_id"].unique())
    assert num_entries == 2*num_exps

    _sort_and_display(df, "seed", "Seeds")
    _sort_and_display(df, "K", "Policy K")
    _sort_and_display(df, "policy_name", "Policy Names")

    if not is_baposgmcp_result:
        _sort_and_display(df, "train_env_name", "Train Envs:")
        _sort_and_display(df, "train_seed", "Train Seeds")
        _sort_and_display(df, "train_alg", "Train Algorithms")

    _sort_and_display(df, "coplayer_K", "Coplayer Policy K")
    _sort_and_display(df, "coplayer_train_seed", "Coplayer Train Seed")

    print("Num rows/entries:", num_entries)
    print("Num experiments:", num_exps)

    if is_baposgmcp_result:
        ba_only_df = df[df["policy_name"] == "BAPOSGMCP_0"]
        print("\nBAPOSGMCP Hyperparameters")
        _sort_and_display(ba_only_df, "num_sims", "Num sims")
        _sort_and_display(ba_only_df, "rollout_K", "Rollout Policy K")
        _sort_and_display(ba_only_df, "rollout_id", "Rollout Policy IDs")
        _sort_and_display(ba_only_df, "rollout_seed", "Rollout Policy Seeds")
        _sort_and_display(ba_only_df, "rollout_alg", "Rollout Policy Algs")
        _sort_and_display(ba_only_df, "other_alg", "Other Agent Policy Algs")
        _sort_and_display(ba_only_df, "other_seed", "Other Agent Policy Seeds")


        for c in BAPOSGMCP_HYPERPARAMETERS:
            values = ba_only_df[c].unique()
            values.sort()
            if len(values) > 1:
                print(f"{c}:", values[0], "+ other values")
            else:
                print(f"{c}:", values[0])


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
        vrange = (-0.2, 1.0)
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
                        row_conds: List,
                        row_seed_key: str,
                        col_conds: List,
                        col_seed_key: str,
                        y_key: str) -> Tuple[List, List, np.ndarray]:
    """Get pairwise values of y variable.

    Returns a 2D np.array where each cell is the value of y variable for
    a given value pairing of (row_seed_key, col_seed_key) variables.

    Also returns ordered list of row and column labels.

    Additionally, constrains each row and col with additional conditions.
    """
    row_df = filter_by(plot_df, row_conds)
    row_seeds = row_df[row_seed_key].unique()
    row_seeds.sort()

    col_df = filter_by(plot_df, col_conds)
    col_seeds = col_df[col_seed_key].unique()
    col_seeds.sort()

    agent_ids = plot_df["agent_id"].unique()
    agent_ids.sort()

    pw_values = np.zeros((len(row_seeds), len(col_seeds)))

    for c, c_seed in enumerate(col_seeds):
        for r, r_seed in enumerate(row_seeds):
            ys = []
            for (a0, a1) in permutations(agent_ids):
                c_seed_conds = [
                    ("agent_id", "==", a0), (col_seed_key, "==", c_seed)
                ]
                c_seed_conds = col_conds + c_seed_conds
                c_seed_df = filter_exps_by(plot_df, c_seed_conds)

                r_seed_conds = [
                    ("agent_id", "==", a1), (row_seed_key, "==", r_seed)
                ]
                r_seed_conds = row_conds + r_seed_conds
                r_seed_df = filter_by(c_seed_df, r_seed_conds)

                y = r_seed_df[y_key].tolist()
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

    return (row_seeds, col_seeds), pw_values


def get_mean_pairwise_values(plot_df,
                             row_conds: List,
                             row_seed_key: str,
                             row_alg_key: str,
                             col_conds: List,
                             col_seed_key: str,
                             col_alg_key: str,
                             y_key: str) -> Tuple[float, float]:
    """Get pairwise mean values of y variable (y_key) over seeds.

    Note this involves taking:
    - for each row_seed - get the average value for each col_seed
    - take average of row_seed averages

    Outputs:
    1. mean for self-play ((row_alg, row_seed) == (col_alg, col_seed)), may be
       np.nan
    2. mean for cross-play
    """
    seeds, pw_returns = get_pairwise_values(
        plot_df,
        row_conds=row_conds,
        row_seed_key=row_seed_key,
        col_conds=col_conds,
        col_seed_key=col_seed_key,
        y_key=y_key
    )

    row_df = filter_by(plot_df, row_conds)
    row_alg = row_df[row_alg_key].unique().tolist()
    assert len(row_alg) == 1
    row_alg = row_alg[0]

    col_df = filter_by(plot_df, col_conds)
    col_alg = col_df[col_alg_key].unique().tolist()
    assert len(col_alg) == 1
    col_alg = col_alg[0]

    xp_values = []
    sp_values = []
    for r, row_seed in enumerate(seeds[0]):
        for c, col_seed in enumerate(seeds[1]):
            v = pw_returns[r][c]
            if np.isnan(v):
                continue
            if (row_alg, row_seed) == (col_alg, col_seed):
                sp_values.append(v)
            else:
                xp_values.append(v)

    if row_alg != col_alg:
        # cross-play only
        return np.nan, np.mean(xp_values)
    return np.mean(sp_values), np.mean(xp_values)

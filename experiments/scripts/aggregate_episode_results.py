"""Script for generating aggregate results from episode results file."""
import os
import argparse
import os.path as osp
from datetime import datetime

import pandas as pd

import baposgmcp.plot as plot_utils
from baposgmcp.run import compile_result_files

# Keys which DF is grouped by
group_keys = [
    "policy_id",
    # "coplayer_policy_id"    # added based on num agents in the env
]

# keys that have constant value across groups
constants = [
    "agent_id",
    "env_id",
    "time_limit",
    "episode_step_limit",
    "discount",
    "c",
    "c_init",
    "c_base",
    "truncated",
    "action_selection",
    "dirichlet_alpha",
    "root_exploration_fraction",
    "reinvigorator",
    "known_bounds",
    "extra_particles_prop",
    "step_limit",
    "epsilon",
    "meta_policy_dict",
    "num_sims",
    "num_episodes",
    "policy_prior_map",
    "fixed_policy_id"
]

replaced = [
    # replaced by number of episodes completed
    "num_episodes",
    # removed/superseded by above
    "episode_number",
    # parsed into num_outcome_...
    "episode_outcome",
    # removed/superseded by 'episode_outcome'
    "episode_done",
]

# take first value in grouped df
first_keys = [
    "exp_id",
    "exp_seed",
]

# values that will be summed across groups
outcome_col_names = ["WIN", "LOSS", "DRAW", "NA"]
sum_keys = outcome_col_names

mean_keys = [
    'search_time',
    'update_time',
    'reinvigoration_time',
    'evaluation_time',
    'policy_calls',
    'inference_time',
    'search_depth',
    'min_value',
    'max_value',
    'episode_return',
    'episode_discounted_return',
    'episode_steps',
    'episode_time'
]


def parse_win(row):  # noqa
    return int(row["episode_outcome"] == 'WIN')


def parse_loss(row):  # noqa
    return int(row["episode_outcome"] == 'LOSS')


def parse_draw(row):  # noqa
    return int(row["episode_outcome"] == 'DRAW')


def parse_na(row):  # noqa
    return int(row["episode_outcome"] not in ('WIN', 'LOSS', 'DRAW'))


def main(parent_dir: str, n_procs: int = 1):   # noqa
    print(f"Compiling results from subdirectories of {parent_dir=}")
    print(f"Saving aggregated results to dir '{parent_dir}'")

    print("Importing data")
    result_filepaths = []
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        for fname in filenames:
            if fname.endswith("episodes.csv"):
                result_filepaths.append(os.path.join(dirpath, fname))
    ep_df = compile_result_files(result_filepaths, verbose=True, n_procs=n_procs)

    print("Adding coplayer policy id column.")
    if len(ep_df["agent_id"].unique().tolist()) == 2:
        ep_df = plot_utils.add_df_coplayer_policy_id(ep_df)
        group_keys.append("coplayer_policy_id")
    else:
        ep_df = plot_utils.add_df_multiple_coplayer_policy_id(ep_df)
        for c in ep_df.columns:
            if c.startswith("coplayer_policy_id"):
                group_keys.append(c)
        group_keys.append("agent_id")
        constants.remove("agent_id")

    print(f"{group_keys=}")

    print("Cleaning data")
    # replace num_episodes with actual number of episodes completed
    ep_df["num_episodes"] = (
        ep_df.groupby(group_keys)["num_episodes"].transform(len)
    )

    # parse episode outcomes into seperate columns
    outcome_col_names = ["WIN", "LOSS", "DRAW", "NA"]
    for k, fn in zip(
        outcome_col_names, [parse_win, parse_loss, parse_draw, parse_na]
    ):
        ep_df[k] = ep_df.apply(fn, axis=1)

    # unassigned keys belong to belief statistics
    assigned_keys = set(
        group_keys + constants + replaced + first_keys + sum_keys + mean_keys
    )
    belief_stat_keys = [c for c in ep_df if c not in assigned_keys]

    print("Grouping data")
    # group by and then aggregate
    gb = ep_df.groupby(group_keys)
    columns = set(list(ep_df.columns))

    agg_dict = {}
    for k in first_keys:
        if k in columns:
            agg_dict[k] = pd.NamedAgg(column=k, aggfunc="min")
        else:
            print(f"Column {k} missing")

    for k in constants:
        if k in columns:
            agg_dict[k] = pd.NamedAgg(column=k, aggfunc="first")
        else:
            print(f"Column {k} missing")

    for k in sum_keys:
        if k in columns:
            agg_dict[f"num_{k}"] = pd.NamedAgg(column=k, aggfunc="sum")
        else:
            print(f"Column {k} missing")

    for k in mean_keys:
        if k in columns:
            agg_dict[f"{k}_mean"] = pd.NamedAgg(column=k, aggfunc="mean")
            agg_dict[f"{k}_std"] = pd.NamedAgg(column=k, aggfunc="std")
            agg_dict[f"{k}_min"] = pd.NamedAgg(column=k, aggfunc="min")
            agg_dict[f"{k}_max"] = pd.NamedAgg(column=k, aggfunc="max")
        else:
            print(f"Column {k} missing")

    for k in belief_stat_keys:
        if k not in columns:
            print(f"Column {k} missing")
        elif k.endswith("_mean") or k.endswith("_std"):
            agg_dict[f"{k}"] = pd.NamedAgg(column=k, aggfunc="mean")
        else:
            agg_dict[f"{k}_mean"] = pd.NamedAgg(column=k, aggfunc="mean")
            agg_dict[f"{k}_std"] = pd.NamedAgg(column=k, aggfunc="std")
            # get count of non nan values since this varies for belief stats
            # based on step number
            agg_dict[f"{k}_n"] = pd.NamedAgg(column=k, aggfunc="count")

    print("Aggregating data")
    gb_agg = gb.agg(**agg_dict)
    compiled_df = gb_agg.reset_index()

    time_str = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"aggregated_results_{time_str}.csv"
    output_filepath = osp.join(parent_dir, output_filename)
    print(f"Saving aggregated results to file '{output_filepath}'")
    compiled_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "parent_dir", type=str,
        help="Path to parent directory."
    )
    parser.add_argument("--n_procs", type=int, default=1)
    main(**vars(parser.parse_args()))

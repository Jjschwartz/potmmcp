"""Functions for getting expected performance (i.e. averaging) results."""
from typing import Sequence

import pandas as pd


def get_uniform_expected_agg_map(df):
    """Get aggregation function map for expected value DF."""
    group_keys = ["policy_id"]

    # take first value in grouped df
    first_keys = [
        "exp_id",
        "exp_seed",
    ]

    # values that will be summed across groups
    sum_keys = [
        k
        for k in df.columns
        if k != "num_sims" and (k.endswith("_n") or k.startswith("num_"))
    ]

    min_keys = [k for k in df.columns if k.endswith("_min")]
    max_keys = [k for k in df.columns if k.endswith("_max")]
    mean_keys = [
        k
        for k in df.columns
        if (
            any(k.endswith(v) for v in ["_mean", "_std", "_CI"])
            or any(
                k.startswith(v)
                for v in ["prop_", "bayes_accuracy", "action_dist_distance"]
            )
        )
    ]

    assigned_keys = set(
        group_keys + first_keys + sum_keys + min_keys + max_keys + mean_keys
    )
    # keys that have constant value across groups
    constants = [k for k in df.columns if k not in assigned_keys]

    columns = set(list(df.columns))

    agg_dict = {}
    for key_list, aggfunc in [
        (first_keys, "min"),
        (constants, "first"),
        (sum_keys, "sum"),
        (min_keys, "min"),
        (max_keys, "max"),
        # TODO change this to weighted mean for non-uniform prior
        (mean_keys, "mean"),
    ]:
        for k in key_list:
            if k in columns:
                agg_dict[k] = pd.NamedAgg(column=k, aggfunc=aggfunc)
            else:
                print(f"Column {k} missing")

    return agg_dict


def get_uniform_expected_df(
    df,
    coplayer_policies: Sequence[str],
    coplayer_policy_key: str = "coplayer_policy_id",
):
    """Get DF with expected values w.r.t policy prior for each policy."""
    agg_dict = get_uniform_expected_agg_map(df)

    exp_df = df[df[coplayer_policy_key].isin(coplayer_policies)]
    gb = exp_df.groupby(["policy_id"])
    gb_agg = gb.agg(**agg_dict)

    print("Ungrouped size =", len(exp_df))
    exp_df = gb_agg.reset_index()
    print("Grouped size =", len(exp_df))

    new_policies = set(exp_df["policy_id"].unique().tolist())
    assert len(new_policies) == len(exp_df), "Should be one row per policy ID"
    return exp_df

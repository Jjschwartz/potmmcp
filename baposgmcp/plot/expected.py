import numpy as np
import matplotlib.pyplot as plt

from baposgmcp.plot.pairwise import get_pairwise_values


def get_expected_values_by_prior(plot_df,
                                 y_key: str,
                                 y_err_key: str,
                                 policy_key: str,
                                 policy_prior,
                                 other_agent_id: int = 1):
    """Get expected value w.r.t policy prior for each policy."""
    pw_values, policy_ids = get_pairwise_values(
        plot_df,
        y_key=y_key,
        policy_key=policy_key,
        average_duplicates=True,
        duplicate_warning=False
    )
    pw_err_values, _ = get_pairwise_values(
        plot_df,
        y_key=y_err_key,
        policy_key=policy_key,
        average_duplicates=True,
        duplicate_warning=False
    )

    expected_values = np.zeros(len(policy_ids))
    expected_err_values = np.zeros(len(policy_ids))
    for i, policy_id in enumerate(policy_ids):
        value = 0.0
        err_value = 0.0
        for coplayer_policy_id, prob in policy_prior[other_agent_id].items():
            coplayer_idx = policy_ids.index(coplayer_policy_id)
            value += pw_values[i][coplayer_idx] * prob
            err_value += pw_err_values[i][coplayer_idx] * prob
        expected_values[i] = value
        expected_err_values[i] = err_value

    return expected_values, expected_err_values, policy_ids


def plot_expected_values_by_num_sims(y_key: str,
                                     expected_values,
                                     expected_err_values,
                                     policy_ids,
                                     policies_with_sims,
                                     policies_without_sims):
    """Plot expected values by num_sims.

    Assumes policies with sims have IDs that end with "_[num_sims]".
    """
    values_by_policy = {}
    all_num_sims = set()
    for policy_prefix in policies_with_sims:
        values_by_policy[policy_prefix] = {"y": {}, "y_err": {}}
        for i, policy_id in enumerate(policy_ids):
            tokens = policy_id.split("_")
            if len(tokens) == 1 or "_".join(tokens[:-1]) != policy_prefix:
                continue
            num_sims = int(tokens[-1])
            value = expected_values[i]
            err_value = expected_err_values[i]
            values_by_policy[policy_prefix]["y"][num_sims] = value
            values_by_policy[policy_prefix]["y_err"][num_sims] = err_value
            all_num_sims.add(num_sims)

    all_num_sims = list(all_num_sims)
    all_num_sims.sort()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))

    for policy_prefix in policies_with_sims:
        y_map = values_by_policy[policy_prefix]["y"]
        y_err_map = values_by_policy[policy_prefix]["y_err"]
        num_sims = list(y_map)
        num_sims.sort()

        y = np.array([y_map[n] for n in num_sims])
        y_err = np.array([y_err_map[n] for n in num_sims])

        ax.plot(num_sims, y, label=policy_prefix)
        plt.fill_between(num_sims, y-y_err, y+y_err, alpha=0.2)

    for policy_id in policies_without_sims:
        i = policy_ids.index(policy_id)
        value = expected_values[i]
        y_err = expected_err_values[i]

        y = np.full(len(all_num_sims), value)
        ax.plot(num_sims, y, label=policy_id)
        plt.fill_between(num_sims, y-y_err, y+y_err, alpha=0.2)

    ax.set_ylabel(y_key)
    ax.set_xlabel("num sims")
    ax.legend()
    plt.show()


def get_and_plot_expected_values_by_num_sims(plot_df,
                                             y_key: str,
                                             y_err_key: str,
                                             policy_key: str,
                                             policy_prior,
                                             policies_with_sims,
                                             policies_without_sims):
    """Get and then plot expected values."""
    exp_values, exp_err_values, policy_ids = get_expected_values_by_prior(
        plot_df,
        y_key=y_key,
        y_err_key=y_err_key,
        policy_key=policy_key,
        policy_prior=policy_prior
    )
    plot_expected_values_by_num_sims(
        y_key=y_key,
        expected_values=exp_values,
        expected_err_values=exp_err_values,
        policy_ids=policy_ids,
        policies_with_sims=policies_with_sims,
        policies_without_sims=policies_without_sims
    )

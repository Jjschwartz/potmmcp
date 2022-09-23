from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _get_belief_stat_keys(y_key_prefix: str,
                          step_limit: int,
                          other_agent_id: int) -> List[str]:
    return [f"{y_key_prefix}_{other_agent_id}_{t}" for t in range(step_limit)]


def plot_expected_belief_stat_by_step(plot_df,
                                      y_key_prefix: str,
                                      policy_ids: List[List[str]],
                                      policy_prior,
                                      step_limit: int,
                                      other_agent_id: int = 1,
                                      policy_key: str = "policy_id",
                                      coplayer_policy_key="coplayer_policy_id",
                                      y_suffix: str = "mean",
                                      y_err_suffix: str = "CI",
                                      figsize: Tuple[int, int] = (9, 9)):
    """Plot expected value of a belief stat w.r.t policy prior by ep step."""
    fig, axs = plt.subplots(
        nrows=1, ncols=len(policy_ids), figsize=figsize, squeeze=False
    )
    xs = np.arange(0, step_limit)

    y_keys = _get_belief_stat_keys(y_key_prefix, step_limit, other_agent_id)
    for i, ax in enumerate(axs[0]):
        for policy_id in policy_ids[i]:
            y = np.zeros(step_limit)
            y_err = np.zeros(step_limit)

            for cp_policy_id, prob in policy_prior[other_agent_id].items():
                sub_df = plot_df[
                    (plot_df[policy_key] == policy_id)
                    & (plot_df[coplayer_policy_key] == cp_policy_id)
                ]
                for t in range(step_limit):
                    y_t = sub_df[f"{y_keys[t]}_{y_suffix}"]
                    y_err_t = sub_df[f"{y_keys[t]}_{y_err_suffix}"]
                    if len(y_t):
                        y[t] += y_t * prob
                        y_err[t] += y_err_t * prob

            ax.plot(xs, y, label=policy_id)
            ax.fill_between(xs, y-y_err, y+y_err, alpha=0.2)

        ax.set_ylabel(y_key_prefix)
        ax.set_xlabel("step")
        ax.legend()

    plt.show()

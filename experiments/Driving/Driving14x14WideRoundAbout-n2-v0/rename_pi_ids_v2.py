import argparse
import os.path as osp

import pandas as pd


policy_id_renaming = {
    "POMetaRollout_greedy_numsims10": "POMetaRollout_greedy_numsims10_actionselectionpucb_truncatedTrue",
    "POMetaRollout_greedy_numsims100": "POMetaRollout_greedy_numsims100_actionselectionpucb_truncatedTrue",
    "POMetaRollout_greedy_numsims1000": "POMetaRollout_greedy_numsims1000_actionselectionpucb_truncatedTrue",
    "POMetaRollout_greedy_numsims50": "POMetaRollout_greedy_numsims50_actionselectionpucb_truncatedTrue",
    "POMetaRollout_greedy_numsims500": "POMetaRollout_greedy_numsims500_actionselectionpucb_truncatedTrue",

    "POMetaRollout_softmax_numsims10": "POMetaRollout_softmax_numsims10_actionselectionpucb_truncatedTrue",
    "POMetaRollout_softmax_numsims100": "POMetaRollout_softmax_numsims100_actionselectionpucb_truncatedTrue",
    "POMetaRollout_softmax_numsims1000": "POMetaRollout_softmax_numsims1000_actionselectionpucb_truncatedTrue",
    "POMetaRollout_softmax_numsims50": "POMetaRollout_softmax_numsims50_actionselectionpucb_truncatedTrue",
    "POMetaRollout_softmax_numsims500": "POMetaRollout_softmax_numsims500_actionselectionpucb_truncatedTrue",

    "POMetaRollout_uniform_numsims10": "POMetaRollout_uniform_numsims10_actionselectionpucb_truncatedTrue",
    "POMetaRollout_uniform_numsims100": "POMetaRollout_uniform_numsims100_actionselectionpucb_truncatedTrue",
    "POMetaRollout_uniform_numsims1000": "POMetaRollout_uniform_numsims1000_actionselectionpucb_truncatedTrue",
    "POMetaRollout_uniform_numsims50": "POMetaRollout_uniform_numsims50_actionselectionpucb_truncatedTrue",
    "POMetaRollout_uniform_numsims500": "POMetaRollout_uniform_numsims500_actionselectionpucb_truncatedTrue",

    # "POMeta_greedy_numsims10",
    # "POMeta_greedy_numsims100",
    # "POMeta_greedy_numsims1000",
    # "POMeta_greedy_numsims50",
    # "POMeta_greedy_numsims500",

    "baposgmcp_fixed_piklrk0seed0-v0_numsims10": "baposgmcp_fixed_piklrk0seed0-v0_numsims10_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk0seed0-v0_numsims100": "baposgmcp_fixed_piklrk0seed0-v0_numsims100_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk0seed0-v0_numsims1000": "baposgmcp_fixed_piklrk0seed0-v0_numsims1000_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk0seed0-v0_numsims50": "baposgmcp_fixed_piklrk0seed0-v0_numsims50_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk0seed0-v0_numsims500": "baposgmcp_fixed_piklrk0seed0-v0_numsims500_actionselectionpucb_truncatedTrue",

    "baposgmcp_fixed_piklrk1seed0-v0_numsims10": "baposgmcp_fixed_piklrk1seed0-v0_numsims10_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk1seed0-v0_numsims100": "baposgmcp_fixed_piklrk1seed0-v0_numsims100_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk1seed0-v0_numsims1000": "baposgmcp_fixed_piklrk1seed0-v0_numsims1000_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk1seed0-v0_numsims50": "baposgmcp_fixed_piklrk1seed0-v0_numsims50_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk1seed0-v0_numsims500": "baposgmcp_fixed_piklrk1seed0-v0_numsims500_actionselectionpucb_truncatedTrue",

    "baposgmcp_fixed_piklrk2seed0-v0_numsims10": "baposgmcp_fixed_piklrk2seed0-v0_numsims10_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk2seed0-v0_numsims100": "baposgmcp_fixed_piklrk2seed0-v0_numsims100_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk2seed0-v0_numsims1000": "baposgmcp_fixed_piklrk2seed0-v0_numsims1000_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk2seed0-v0_numsims50": "baposgmcp_fixed_piklrk2seed0-v0_numsims50_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk2seed0-v0_numsims500": "baposgmcp_fixed_piklrk2seed0-v0_numsims500_actionselectionpucb_truncatedTrue",

    "baposgmcp_fixed_piklrk3seed0-v0_numsims10": "baposgmcp_fixed_piklrk3seed0-v0_numsims10_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk3seed0-v0_numsims100": "baposgmcp_fixed_piklrk3seed0-v0_numsims100_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk3seed0-v0_numsims1000": "baposgmcp_fixed_piklrk3seed0-v0_numsims1000_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk3seed0-v0_numsims50": "baposgmcp_fixed_piklrk3seed0-v0_numsims50_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk3seed0-v0_numsims500": "baposgmcp_fixed_piklrk3seed0-v0_numsims500_actionselectionpucb_truncatedTrue",

    "baposgmcp_fixed_piklrk4seed0-v0_numsims10": "baposgmcp_fixed_piklrk4seed0-v0_numsims10_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk4seed0-v0_numsims100": "baposgmcp_fixed_piklrk4seed0-v0_numsims100_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk4seed0-v0_numsims1000": "baposgmcp_fixed_piklrk4seed0-v0_numsims1000_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk4seed0-v0_numsims50": "baposgmcp_fixed_piklrk4seed0-v0_numsims50_actionselectionpucb_truncatedTrue",
    "baposgmcp_fixed_piklrk4seed0-v0_numsims500": "baposgmcp_fixed_piklrk4seed0-v0_numsims500_actionselectionpucb_truncatedTrue",

    "baposgmcp_greedy_numsims10": "baposgmcp_greedy_numsims10_actionselectionpucb_truncatedTrue",
    "baposgmcp_greedy_numsims100": "baposgmcp_greedy_numsims100_actionselectionpucb_truncatedTrue",
    "baposgmcp_greedy_numsims1000": "baposgmcp_greedy_numsims1000_actionselectionpucb_truncatedTrue",
    "baposgmcp_greedy_numsims50": "baposgmcp_greedy_numsims500_actionselectionpucb_truncatedTrue",
    "baposgmcp_greedy_numsims500": "baposgmcp_greedy_numsims50_actionselectionpucb_truncatedTrue",

    # "baposgmcp_random_numsims1000_actionselectionpucb_truncatedFalse",
    # "baposgmcp_random_numsims100_actionselectionpucb_truncatedFalse",
    # "baposgmcp_random_numsims10_actionselectionpucb_truncatedFalse",
    # "baposgmcp_random_numsims500_actionselectionpucb_truncatedFalse",
    # "baposgmcp_random_numsims50_actionselectionpucb_truncatedFalse",

    "baposgmcp_softmax_numsims10": "baposgmcp_softmax_numsims10_actionselectionpucb_truncatedTrue",
    "baposgmcp_softmax_numsims100": "baposgmcp_softmax_numsims100_actionselectionpucb_truncatedTrue",
    "baposgmcp_softmax_numsims1000": "baposgmcp_softmax_numsims1000_actionselectionpucb_truncatedTrue",
    "baposgmcp_softmax_numsims50": "baposgmcp_softmax_numsims50_actionselectionpucb_truncatedTrue",
    "baposgmcp_softmax_numsims500": "baposgmcp_softmax_numsims500_actionselectionpucb_truncatedTrue",
    "baposgmcp_uniform_numsims10": "baposgmcp_uniform_numsims10_actionselectionpucb_truncatedTrue",
    "baposgmcp_uniform_numsims100": "baposgmcp_uniform_numsims100_actionselectionpucb_truncatedTrue",
    "baposgmcp_uniform_numsims1000": "baposgmcp_uniform_numsims1000_actionselectionpucb_truncatedTrue",
    "baposgmcp_uniform_numsims50": "baposgmcp_uniform_numsims50_actionselectionpucb_truncatedTrue",
    "baposgmcp_uniform_numsims500": "baposgmcp_uniform_numsims500_actionselectionpucb_truncatedTrue",
}


def rename_policy_id(row):   # noqa
    pi_id = row["policy_id"]
    return policy_id_renaming.get(pi_id, pi_id)


def main(results_filepath):   # noqa
    results_dir = osp.dirname(results_filepath)

    print("Importing data")
    ep_df = pd.read_csv(results_filepath)

    print("Renaming policy IDs")
    ep_df["policy_id"] = ep_df.apply(rename_policy_id, axis=1)

    print("Writing to file")
    ep_df.to_csv(
        osp.join(results_dir, "compiled_episode_results_rename.csv"),
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "results_filepath", type=str,
        help="Path to episode results file."
    )
    main(**vars(parser.parse_args()))

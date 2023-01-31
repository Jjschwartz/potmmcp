"""Script for replacing old UCBMCP results with new ones."""
import argparse
import os
import os.path as osp
import shutil

import pandas as pd


new_dir_prefix = "ucbmcp_exp"
old_dir_prefix = "exp"


def get_exp_map(parent_dir: str, exp_dir: str):
    """Get map from (policy_id_0, policy_id_1, ...) to exp_id and exp filepaths."""
    # get actual results dir <result_dir>/<new_dir>/<baposgmcp_init...>
    assert len(os.listdir(osp.join(parent_dir, exp_dir))) == 1
    sub_dir = os.listdir(osp.join(parent_dir, exp_dir))[0]

    # get map from (policy_id_0, policy_id_1, ...) to exp_id and files
    exp_map = {}
    for dirpath, dirnames, filenames in os.walk(osp.join(parent_dir, exp_dir, sub_dir)):
        for fname in filenames:
            if not (
                fname.startswith("exp")
                and fname.endswith(".csv")
                and not fname.endswith("_episodes.csv")
            ):
                continue
            exp_id = int(fname.split(".")[0].split("_")[1])

            exp_df = pd.read_csv(osp.join(dirpath, fname))
            agent_ids = exp_df["agent_id"].unique().tolist()
            agent_ids.sort()

            policy_ids = [
                exp_df[exp_df["agent_id"] == i]["policy_id"].unique().tolist()
                for i in agent_ids
            ]
            assert all(
                len(pi_ids) == 1 for pi_ids in policy_ids
            ), f"More than one policy for an agent: {policy_ids}"
            policy_ids = [pi_ids[0] for pi_ids in policy_ids]
            exp_map[tuple(policy_ids)] = {
                "exp_id": exp_id,
                "fpaths": {
                    "csv": osp.join(dirpath, fname),
                    "episodes.csv": osp.join(dirpath, f"exp_{exp_id}_episodes.csv"),
                    "log": osp.join(dirpath, f"exp_{exp_id}.log"),
                },
            }

    return exp_map


def do_things(results_dir: str):
    output_dir = osp.join(results_dir, "replaced_results")

    print("Creating new output dir")
    try:
        os.makedirs(output_dir, exist_ok=False)
    except OSError:
        print("Deleting and replacing existing output dir")
        # make clean dir
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=False)

    # need to replace the exp_X.csv, exp_X_episodes.csv, exp_X.log in the old dir
    # with the corresponding exp_Y.csv, exp_Y_episodes.csv, exp_Y.log from the new dir
    # and change the "exp_id" value from Y to X in the new ,csv files (ignore the log)

    # start by getting all the old and new dirs
    old_dirs = []
    new_dirs = []
    for exp_dir in os.listdir(results_dir):
        if exp_dir.startswith(old_dir_prefix):
            old_dirs.append(exp_dir)
        elif exp_dir.startswith(new_dir_prefix):
            new_dirs.append(exp_dir)
        else:
            print(f"Ignoring exp dir: {exp_dir}")

    print("old_dirs:", old_dirs)
    print("new_dirs:", new_dirs)

    # Next we keep only old dirs that match a new dir seed
    # and we copy these old dirs to the output dir
    # (this will be the version we update, leaving the original version unchanged)
    new_old_dirs_map = {}
    for new_dir in new_dirs:
        seed = [s for s in new_dir.split("_") if s.startswith("seed")][0]
        old_dir = [d for d in old_dirs if any(s == seed for s in d.split("_"))]
        if len(old_dir) == 0:
            print(f"No matching exp for {seed}")
            continue
        old_dir = old_dir[0]
        print("Matching experiment seed:", new_dir, old_dir)
        new_old_dirs_map[new_dir] = old_dir

        # copy old dir to output parent dir
        shutil.copytree(osp.join(results_dir, old_dir), osp.join(output_dir, old_dir))


    for new_dir in new_old_dirs_map:
        old_dir = new_old_dirs_map[new_dir]
        print(f"Replacing exps from {old_dir} with {new_dir}")
        new_exp_map = get_exp_map(results_dir, new_dir)
        old_exp_map = get_exp_map(output_dir, old_dir)

        # for each new exp
        for policy_ids, new_exp_info in new_exp_map.items():
            old_exp_info = old_exp_map[policy_ids]
            old_exp_id = old_exp_info["exp_id"]
            # replace each old exp file with the new exp file
            for ext in new_exp_info["fpaths"]:
                new_file = new_exp_info['fpaths'][ext]
                old_file = old_exp_info['fpaths'][ext]
                shutil.copy(new_file, old_file)

                # update the exp_id within the copied new file (i.e. now at location
                # of old file) to match the exp id in the OG old file
                if ext in ("csv", "episodes.csv"):
                    exp_df = pd.read_csv(old_file)
                    exp_df["exp_id"] = old_exp_id
                    exp_df.to_csv(old_file, index=False)

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to parent directory containing old and new experimet result dirs .",
    )
    do_things(**vars(parser.parse_args()))

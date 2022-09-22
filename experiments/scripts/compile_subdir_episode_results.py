"""Compiles exp_[ID]_episodes.csv files in subdirectories into single file."""
import os
import argparse

from baposgmcp.run import compile_result_files


def _main(args):
    parent_dir = args.parent_dirpath
    print(f"Compiling results from subdirectories of {parent_dir=}")

    child_dirpaths = [
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    print(f"Compiling results from {len(child_dirpaths)} subdirectories.")

    result_filepaths = []
    for child_dir in child_dirpaths:
        for file_name in os.listdir(child_dir):
            if file_name.endswith("episodes.csv"):
                child_resultpath = os.path.join(child_dir, file_name)
                result_filepaths.append(child_resultpath)

    print(f"Compiling results from {len(result_filepaths)} results files.")
    result_filepath = compile_result_files(
        parent_dir,
        result_filepaths,
        compiled_results_filename="compiled_episode_resuls.csv"
    )
    print(f"Results compiled. Results file={result_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "parent_dirpath", type=str,
        help="Path to parent directory."
    )
    _main(parser.parse_args())

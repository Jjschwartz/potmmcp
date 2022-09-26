"""Compiles exp_[ID]_episodes.csv files in subdirectories into single file."""
import os
import argparse

from baposgmcp.run import compile_result_files


def _main(args):
    parent_dir = args.parent_dirpath
    print(f"Compiling results from subdirectories of {parent_dir=}")

    result_filepaths = []
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        for fname in filenames:
            if fname.endswith("episodes.csv"):
                result_filepaths.append(os.path.join(dirpath, fname))

    print(f"Compiling results from {len(result_filepaths)} results files.")
    result_filepath = compile_result_files(
        parent_dir,
        result_filepaths,
        compiled_results_filename="compiled_episode_results.csv"
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

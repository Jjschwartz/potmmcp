"""Compiles exp_[ID]_episodes.csv files in subdirectories into single file."""
import os
import argparse

from baposgmcp.run import compile_and_save_result_files


def compile_episode_results(parent_dir: str, n_procs: int = 1):
    print(f"Compiling results from subdirectories of {parent_dir=}")

    result_filepaths = []
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        for fname in filenames:
            if fname.endswith("episodes.csv"):
                result_filepaths.append(os.path.join(dirpath, fname))

    print(f"Compiling results from {len(result_filepaths)} results files.")
    result_filepath = compile_and_save_result_files(
        parent_dir,
        result_filepaths,
        compiled_results_filename="compiled_episode_results.csv",
        verbose=True,
        n_procs=n_procs
    )
    print(f"Results compiled. Results file={result_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "parent_dir", type=str,
        help="Path to parent directory."
    )
    parser.add_argument("--n_procs", type=int, default=1)
    compile_episode_results(**vars(parser.parse_args()))

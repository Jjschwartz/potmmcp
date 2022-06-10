"""Compiles compiled_results.csv files in subdirectories into a single file."""
import os
import argparse

from baposgmcp.stats import (
    compile_results, compile_result_files, COMPILED_RESULTS_FNAME
)


def _main(args):
    parent_dir = args.parent_dirpath
    print("Compiling results from subdirectories of {parent_dir=}")

    child_dirpaths = [
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]

    result_filepaths = []
    for child_dir in child_dirpaths:
        child_resultpath = os.path.join(child_dir, COMPILED_RESULTS_FNAME)
        file_exists = os.path.exists(child_resultpath)
        if not file_exists and not args.skip_handle_missing:
            child_resultpath = compile_results(child_dir)
        elif not file_exists:
            continue
        result_filepaths.append(child_resultpath)

    result_filepath = compile_result_files(parent_dir, result_filepaths)
    print(f"Results compiled. Results file={result_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "parent_dirpath", type=str,
        help="Path to parent directory."
    )
    parser.add_argument(
        "--skip_handle_missing", action="store_true",
        help="Skip attempts to generate mising 'compiled_results.csv' files."
    )
    _main(parser.parse_args())

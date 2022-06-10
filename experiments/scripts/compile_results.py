"""Compiles all .csv files in a directory into a single file."""
import argparse

from baposgmcp.stats import compile_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "result_dir_path", type=str,
        help="Path to directory containing files to compile."
    )
    parser.add_argument(
        "--extra_output_dir", type=str, default=None,
        help="Optional path to directory to save output to."
    )
    parser.add_argument(
        "--recurse", action="store_True",
        help="Recurse into sub-directories."
    )
    args = parser.parse_args()
    compile_results(args.result_dir_path, args.extra_output_dir)

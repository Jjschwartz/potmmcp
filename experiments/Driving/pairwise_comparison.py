"""Script for running pairwise evaluation of trained Rllib policies.

The script takes a list of environments names and a list of rllib policy save
directories as arguments. It then runs a pairwise evaluation between each
policy in each of the policy directories for each environment.

"""
from pprint import pprint

import baposgmcp.run as run_lib
import baposgmcp.rllib as ba_rllib


def main(args):    # noqa
    ba_rllib.register_posggym_env(args.env_name)

    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    exp_params_list = run_lib.get_rl_exp_params(**vars(args))

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_name = f"pairwise_comparison_{seed_str}"

    print(f"== Running {len(exp_params_list)} Experiments ==")
    print(f"== Using {args.n_procs} CPUs ==")
    run_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        exp_args=vars(args)
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = run_lib.get_rl_exp_parser()
    main(parser.parse_args())

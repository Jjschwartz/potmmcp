"""Script for running pairwise evaluation of posggym policies.

The script takes an environment ID and a lists of policy ids as arguments.
It then runs a pairwise evaluation for each possible pairing of policies.

"""
from pprint import pprint

import posggym_agents.exp as exp_lib


def main(args):    # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    exp_params_list = exp_lib.get_asymmetric_pairwise_exp_params(**vars(args))

    exp_name = f"pairwise_initseed{args.init_seed}_numseeds{args.num_seeds}"

    print(f"== Running {len(exp_params_list)} Experiments ==")
    print(f"== Using {args.n_procs} CPUs ==")
    exp_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        exp_args=vars(args)
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = exp_lib.get_asymmetric_pairwise_exp_parser()
    main(parser.parse_args())

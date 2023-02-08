"""Run BAPOSGMCP with the different meta-policies in Driving env with KLR policies.

Experiments
- BAPOSGMCP (PUCB + Meta-Policy) - best [Truncated]
- BAPOSGMCP-random
- IPOMCP (UCB + Meta-Policy) - Best meta-policy [Truncated]
- IPOMCP Random (UCB + Random) [Untruncated]
- Meta-Policy - best

"""
import copy
from pprint import pprint

from run_experiment import (
    BAPOSGMCP_PUCT_KWARGS,
    DISCOUNT,
    ENV_ID,
    ENV_STEP_LIMIT,
    GREEDY_META_POLICY_MAP,
    N_AGENTS,
    OTHER_AGENT_ID,
    POLICY_IDS,
    POLICY_PRIOR_MAP,
    SOFTMAX_META_POLICY_MAP,
    UNIFORM_META_POLICY_MAP,
)

import baposgmcp.baselines as baseline_lib
import baposgmcp.run as run_lib


SEARCH_TIME_LIMITS = [0.1, 1, 5, 10, 20]

# debug
# SEARCH_TIME_LIMITS = [0.05]


def get_baposgmcps():  # noqa
    kwargs = copy.deepcopy(BAPOSGMCP_PUCT_KWARGS)
    kwargs["num_sims"] = None

    variable_params = {"search_time_limit": SEARCH_TIME_LIMITS, "truncated": [True]}

    baposgmcp_params = []
    for (name, meta_policy_map) in [
        ("greedy", GREEDY_META_POLICY_MAP),
        ("softmax", SOFTMAX_META_POLICY_MAP),
        ("uniform", UNIFORM_META_POLICY_MAP),
    ]:
        baposgmcp_params.extend(
            run_lib.load_baposgmcp_params(
                variable_params=variable_params,
                baposgmcp_kwargs=kwargs,
                policy_prior_map=POLICY_PRIOR_MAP,
                meta_policy_dict=meta_policy_map,
                base_policy_id=f"baposgmcp_meta{name}",
            )
        )

    random_kwargs = copy.deepcopy(kwargs)
    random_kwargs["truncated"] = False
    baposgmcp_params.extend(
        baseline_lib.load_random_baposgmcp_params(
            variable_params={
                "search_time_limit": SEARCH_TIME_LIMITS,
                "truncated": [False],
            },
            baposgmcp_kwargs=random_kwargs,
            policy_prior_map=POLICY_PRIOR_MAP,
            base_policy_id="baposgmcp-random",
        )
    )

    baposgmcp_params.extend(
        baseline_lib.load_fixed_pi_baposgmcp_params(
            variable_params=variable_params,
            fixed_policy_ids=POLICY_IDS,
            baposgmcp_kwargs=kwargs,
            policy_prior_map=POLICY_PRIOR_MAP,
            base_policy_id="baposgmcp-fixed",
        )
    )

    # NUM Exps:
    # = |SEARCH_TIME_LIMITS| * (|Meta| + 1 + |PIS|)
    # = 5 * (3 + 5 + 1)
    # = 45
    assert len(baposgmcp_params) == len(SEARCH_TIME_LIMITS) * (3 + 1 + len(POLICY_IDS))
    return baposgmcp_params


def main(args):  # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    other_params = run_lib.load_posggym_agent_params(
        list(POLICY_PRIOR_MAP[OTHER_AGENT_ID])
    )
    assert len(other_params) == len(POLICY_PRIOR_MAP[OTHER_AGENT_ID])
    print(f"Num Other params = {len(other_params)}")

    # - BAPOSGMCP (PUCB + Meta-Policy)
    #   - Greedy [Truncated]
    #   - Softmax [Truncated]
    #   - Uniform [Truncated]
    # - BAPOSGMCP-random
    # - BAPOSGMCP (PUCB + Fixed policies) [Truncated]
    policy_params = get_baposgmcps()
    print(f"Num policy_params={len(policy_params)}")

    exp_params_list = run_lib.get_pairwise_exp_params(
        ENV_ID,
        [policy_params, other_params],
        discount=DISCOUNT,
        exp_id_init=0,
        tracker_fn=run_lib.belief_tracker_fn,
        tracker_fn_kwargs={
            "num_agents": N_AGENTS,
            "step_limit": ENV_STEP_LIMIT,
            "discount": DISCOUNT,
        },
        renderer_fn=None,
        **vars(args),
    )

    if args.get_num_exps:
        print(f"Number of experiments = {len(exp_params_list)}")
        return

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_str = "" if args.run_exp_id is None else f"_exp{args.run_exp_id}"
    exp_name = f"baposgmcp{exp_str}_{seed_str}"

    exp_args = vars(args)
    exp_args["env_id"] = ENV_ID
    exp_args["discount"] = DISCOUNT
    exp_args["baposgmcp_kwargs"] = BAPOSGMCP_PUCT_KWARGS

    print(f"== Using {args.n_procs} CPUs ==")
    run_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        exp_args=vars(args),
        root_save_dir=args.root_save_dir,
        run_exp_id=args.run_exp_id,
    )
    print("== All done ==")


if __name__ == "__main__":
    parser = run_lib.get_exp_parser()
    parser.add_argument(
        "--get_num_exps",
        action="store_true",
        help="Compute and display number of experiments without running them.",
    )
    main(parser.parse_args())

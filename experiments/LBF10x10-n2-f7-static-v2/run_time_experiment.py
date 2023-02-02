"""Run BAPOSGMCP experiment in LBF env with heuristic policies."""
import copy
from pprint import pprint

from run_experiment import (
    BAPOSGMCP_PUCT_KWARGS,
    BEST_META_PI_MAP,
    BEST_META_PI_NAME,
    DISCOUNT,
    ENV_ID,
    ENV_STEP_LIMIT,
    N_AGENTS,
    POLICY_IDS,
    POLICY_PRIOR_MAP,
    UCB_C,
    get_metabaseline,
)

import baposgmcp.baselines as baseline_lib
import baposgmcp.run as run_lib


SEARCH_TIME_LIMITS = [0.1, 1, 5, 10, 20]

# debug
# SEARCH_TIME_LIMITS = [0.05]


def get_baposgmcps():  # noqa
    kwargs = copy.deepcopy(BAPOSGMCP_PUCT_KWARGS)
    kwargs["num_sims"] = None

    baposgmcp_params = run_lib.load_baposgmcp_params(
        variable_params={"search_time_limit": SEARCH_TIME_LIMITS, "truncated": [False]},
        baposgmcp_kwargs=kwargs,
        policy_prior_map=POLICY_PRIOR_MAP,
        meta_policy_dict=BEST_META_PI_MAP,
        base_policy_id=f"baposgmcp_meta{BEST_META_PI_NAME}",
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

    # NUM Exps:
    # = |SEARCH_TIME_LIMITS| * 2
    # = 10
    assert len(baposgmcp_params) == len(SEARCH_TIME_LIMITS) * 2
    return baposgmcp_params


def get_ucb_mcps():  # noqa
    """
    - UCB MCP (UCB + Meta-Policy)
      - Best meta-policy [Truncated]
    - UCB MCP Random (UCB + Random) [Untruncated]
    """
    ucb_kwargs = copy.deepcopy(BAPOSGMCP_PUCT_KWARGS)
    ucb_kwargs["num_sims"] = None
    ucb_kwargs["c"] = UCB_C
    ucb_kwargs["action_selection"] = "ucb"

    ucb_params = run_lib.load_baposgmcp_params(
        variable_params={"search_time_limit": SEARCH_TIME_LIMITS, "truncated": [False]},
        baposgmcp_kwargs=ucb_kwargs,
        policy_prior_map=POLICY_PRIOR_MAP,
        meta_policy_dict=BEST_META_PI_MAP,
        base_policy_id=f"ucbmcp_meta{BEST_META_PI_NAME}",
    )

    random_kwargs = copy.deepcopy(ucb_kwargs)
    random_kwargs["truncated"] = False
    ucb_params.extend(
        baseline_lib.load_random_baposgmcp_params(
            variable_params={
                "search_time_limit": SEARCH_TIME_LIMITS,
                "truncated": [False],
            },
            baposgmcp_kwargs=random_kwargs,
            policy_prior_map=POLICY_PRIOR_MAP,
            base_policy_id="ucbmcp-random",
        )
    )

    # NUM Exps:
    # = |SEARCH_TIME_LIMITS| + |SEARCH_TIME_LIMITS|
    # = 5 + 5
    # = 10
    assert len(ucb_params) == len(SEARCH_TIME_LIMITS) * 2
    return ucb_params


def main(args):  # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    other_params = run_lib.load_posggym_agent_params(POLICY_IDS)
    print(f"Num Other params = {len(other_params)}")

    # - BAPOSGMCP (PUCB + Meta-Policy)
    #   - best [Truncated]
    # - BAPOSGMCP-random
    policy_params = get_baposgmcps()
    # - IPOMCP (UCB + Meta-Policy)
    #   - Best meta-policy [Truncated]
    # - IPOMCP Random (UCB + Random) [Untruncated]
    policy_params.extend(get_ucb_mcps())
    # - Meta-Policy
    #   - best
    policy_params.extend(get_metabaseline(best_only=True))
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
    exp_args["policy_ids"] = POLICY_IDS
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

"""Run BAPOSGMCP experiment in LBF env with heuristic policies."""
import copy
from pprint import pprint

import baposgmcp.baselines as baseline_lib
import baposgmcp.run as run_lib
from baposgmcp import meta_policy


ENV_ID = "LBF10x10-n2-f7-static-v2"
N_AGENTS = 2
ENV_STEP_LIMIT = 50

DISCOUNT = 0.99
NUM_SIMS = [10, 50, 100, 500, 1000]
BAPOSGMCP_AGENT_ID = 0
OTHER_AGENT_ID = 1

# DEBUG - Delete/comment this
# NUM_SIMS = [2, 5]

POLICY_IDS = [
    f"{ENV_ID}/heuristic1-v0",
    f"{ENV_ID}/heuristic2-v0",
    f"{ENV_ID}/heuristic3-v0",
    f"{ENV_ID}/heuristic4-v0",
]
POLICY_PRIOR_MAP = {
    OTHER_AGENT_ID: {
        f"{ENV_ID}/heuristic1-v0": 1 / 4,
        f"{ENV_ID}/heuristic2-v0": 1 / 4,
        f"{ENV_ID}/heuristic3-v0": 1 / 4,
        f"{ENV_ID}/heuristic4-v0": 1 / 4,
    }
}
PAIRWISE_RETURNS = {
    (-1, f"{ENV_ID}/heuristic1-v0"): {
        f"{ENV_ID}/heuristic1-v0": 0.39,
        f"{ENV_ID}/heuristic2-v0": 0.04,
        f"{ENV_ID}/heuristic3-v0": 0.28,
        f"{ENV_ID}/heuristic4-v0": 0.04,
    },
    (-1, f"{ENV_ID}/heuristic2-v0"): {
        f"{ENV_ID}/heuristic1-v0": 0.55,
        f"{ENV_ID}/heuristic2-v0": 0.05,
        f"{ENV_ID}/heuristic3-v0": 0.47,
        f"{ENV_ID}/heuristic4-v0": 0.05,
    },
    (-1, f"{ENV_ID}/heuristic3-v0"): {
        f"{ENV_ID}/heuristic1-v0": 0.51,
        f"{ENV_ID}/heuristic2-v0": 0.10,
        f"{ENV_ID}/heuristic3-v0": 0.40,
        f"{ENV_ID}/heuristic4-v0": 0.04,
    },
    (-1, f"{ENV_ID}/heuristic4-v0"): {
        f"{ENV_ID}/heuristic1-v0": 0.55,
        f"{ENV_ID}/heuristic2-v0": 0.06,
        f"{ENV_ID}/heuristic3-v0": 0.60,
        f"{ENV_ID}/heuristic4-v0": 0.05,
    },
}

GREEDY_META_POLICY_MAP = meta_policy.get_greedy_policy_dict(PAIRWISE_RETURNS)
UNIFORM_META_POLICY_MAP = meta_policy.get_uniform_policy_dict(PAIRWISE_RETURNS)
SOFTMAX_META_POLICY_MAP = meta_policy.get_softmax_policy_dict(PAIRWISE_RETURNS, 0.25)

BEST_META_PI_NAME = "uniform"
BEST_META_PI_MAP = UNIFORM_META_POLICY_MAP

BAPOSGMCP_PUCT_KWARGS = {
    "discount": DISCOUNT,
    "c_init": 1.25,
    "c_base": 20000,
    "truncated": False,
    "action_selection": "pucb",
    "dirichlet_alpha": 0.6,  # 6 actions / 10
    "root_exploration_fraction": 0.5,  # half actions valid/useful at any step
    "reinvigorator": None,  # Use default rejection sampler
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": ENV_STEP_LIMIT,
    "epsilon": 0.01,
}
UCB_C = 2.0  # as per OG paper/standard parameter


def get_metabaseline():  # noqa
    baseline_params = []
    for (name, meta_policy_map) in [
        ("greedy", GREEDY_META_POLICY_MAP),
        ("softmax", SOFTMAX_META_POLICY_MAP),
        ("uniform", UNIFORM_META_POLICY_MAP),
    ]:
        # Meta Baseline Policy
        policy_id = f"metabaseline_{name}"
        policy_params = run_lib.PolicyParams(
            id=policy_id,
            entry_point=baseline_lib.MetaBaselinePolicy.posggym_agents_entry_point,
            kwargs={
                "policy_id": policy_id,
                "policy_prior_map": POLICY_PRIOR_MAP,
                "meta_policy_dict": meta_policy_map,
            },
        )
        baseline_params.append(policy_params)

    # Num exps:
    # = |Meta|
    # = 3
    n_meta = 3
    assert len(baseline_params) == n_meta
    return baseline_params


def get_baposgmcps():  # noqa
    variable_params = {"num_sims": NUM_SIMS, "truncated": [False]}

    baposgmcp_params = []
    for (name, meta_policy_map) in [
        ("greedy", GREEDY_META_POLICY_MAP),
        ("softmax", SOFTMAX_META_POLICY_MAP),
        ("uniform", UNIFORM_META_POLICY_MAP),
    ]:
        baposgmcp_params.extend(
            run_lib.load_baposgmcp_params(
                variable_params=variable_params,
                baposgmcp_kwargs=BAPOSGMCP_PUCT_KWARGS,
                policy_prior_map=POLICY_PRIOR_MAP,
                meta_policy_dict=meta_policy_map,
                base_policy_id=f"baposgmcp_meta{name}",
            )
        )
    # NUM Exps:
    # = |NUM_SIMS| * |Meta|
    # = 5 * 3
    # = 15
    assert len(baposgmcp_params) == (len(NUM_SIMS) * 3)
    return baposgmcp_params


def get_fixed_baposgmcps():  # noqa
    random_variable_params = {
        "num_sims": NUM_SIMS,
        "truncated": [False],  # added so it's clearly visible in policy id
    }
    random_kwargs = copy.deepcopy(BAPOSGMCP_PUCT_KWARGS)
    random_kwargs["truncated"] = False
    baposgmcp_params = baseline_lib.load_random_baposgmcp_params(
        variable_params=random_variable_params,
        baposgmcp_kwargs=random_kwargs,
        policy_prior_map=POLICY_PRIOR_MAP,
        base_policy_id="baposgmcp-random",
    )

    fixed_variable_params = {"num_sims": NUM_SIMS, "truncated": [False]}
    baposgmcp_params.extend(
        baseline_lib.load_fixed_pi_baposgmcp_params(
            variable_params=fixed_variable_params,
            fixed_policy_ids=POLICY_IDS,
            baposgmcp_kwargs=BAPOSGMCP_PUCT_KWARGS,
            policy_prior_map=POLICY_PRIOR_MAP,
            base_policy_id="baposgmcp-fixed",
        )
    )
    # NUM Exps:
    # = |NUM_SIMS| * (|PIS| + 1)
    # = 5 * (4 + 1)
    # = 25
    assert len(baposgmcp_params) == (len(NUM_SIMS) * 1 * 1 * (len(POLICY_IDS) + 1))
    return baposgmcp_params


def get_ucb_mcps():  # noqa
    """
    - IPOMCP (UCB + Meta-Policy)
      - Best meta-policy [Truncated]
      - Best meta-policy [UnTruncated]
    - IPOMCP Random (UCB + Random) [Untruncated]
    """
    ucb_kwargs = copy.deepcopy(BAPOSGMCP_PUCT_KWARGS)
    ucb_kwargs["c_init"] = UCB_C
    ucb_kwargs["action_selection"] = "ucb"

    meta_variable_params = {"num_sims": NUM_SIMS, "truncated": [False]}
    ucb_params = run_lib.load_baposgmcp_params(
        variable_params=meta_variable_params,
        baposgmcp_kwargs=ucb_kwargs,
        policy_prior_map=POLICY_PRIOR_MAP,
        meta_policy_dict=BEST_META_PI_MAP,
        base_policy_id=f"ucbmcp_meta{BEST_META_PI_NAME}",
    )

    random_variable_params = {"num_sims": NUM_SIMS, "truncated": [False]}
    random_kwargs = copy.deepcopy(ucb_kwargs)
    random_kwargs["truncated"] = False
    ucb_params.extend(
        baseline_lib.load_random_baposgmcp_params(
            variable_params=random_variable_params,
            baposgmcp_kwargs=random_kwargs,
            policy_prior_map=POLICY_PRIOR_MAP,
            base_policy_id="ucbmcp-random",
        )
    )

    # NUM Exps:
    # = |NUM_SIMS| * |TRUNCATED| + |NUM_SIMS|
    # = 5 * 1 + 5
    # = 10
    assert len(ucb_params) == len(NUM_SIMS) * 2
    return ucb_params


def main(args):  # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    other_params = run_lib.load_posggym_agent_params(POLICY_IDS)

    # All untruncated since LBF heuristic policies don't have value functions
    # - BAPOSGMCP (PUCB + Meta-Policy)
    #   - Greedy
    #   - Softmax
    #   - Uniform
    policy_params = get_baposgmcps()
    # - BAPOSGMCP (PUCB + Random)
    # - BAPOSGMCP (PUCB + Fixed policies)
    policy_params.extend(get_fixed_baposgmcps())
    # - UCB MCP (UCB + Meta-Policy)
    #   - Best meta-policy
    # - UCB MCP Random (UCB + Random)
    policy_params.extend(get_ucb_mcps())
    # - Meta-Policy
    #   - Greedy
    #   - Softmax
    #   - Uniform
    policy_params.extend(get_metabaseline())

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
    exp_name = f"baposgmcp_heuristic{exp_str}_{seed_str}"

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

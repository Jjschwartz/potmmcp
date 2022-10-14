"""Run BAPOSGMCP experiment in LBF env with heuristic policies."""
from pprint import pprint

import baposgmcp.run as run_lib
from baposgmcp import meta_policy
import baposgmcp.baselines as baseline_lib


ENV_ID = "LBF5x5-n2-f3-static-v2"
N_AGENTS = 2
ENV_STEP_LIMIT = 50

DISCOUNT = 0.99
NUM_SIMS = [10, 50, 100, 500, 1000, 2000]
BAPOSGMCP_AGENT_ID = 0
OTHER_AGENT_ID = 1

POLICY_IDS = [
    f"{ENV_ID}/heuristic1-v0",
    f"{ENV_ID}/heuristic2-v0",
    f"{ENV_ID}/heuristic3-v0",
    f"{ENV_ID}/heuristic4-v0"
]
POLICY_PRIOR_MAP = {OTHER_AGENT_ID: {
    f"{ENV_ID}/heuristic1-v0": 1/4,
    f"{ENV_ID}/heuristic2-v0": 1/4,
    f"{ENV_ID}/heuristic3-v0": 1/4,
    f"{ENV_ID}/heuristic4-v0": 1/4
}}

PAIRWISE_RETURNS = {
    (-1, f"{ENV_ID}/heuristic1-v0"): {
        f"{ENV_ID}/heuristic1-v0": 0.48,
        f"{ENV_ID}/heuristic2-v0": 0.28,
        f"{ENV_ID}/heuristic3-v0": 0.34,
        f"{ENV_ID}/heuristic4-v0": 0.28
    },
    (-1, f"{ENV_ID}/heuristic2-v0"): {
        f"{ENV_ID}/heuristic1-v0": 0.69,
        f"{ENV_ID}/heuristic2-v0": 0.48,
        f"{ENV_ID}/heuristic3-v0": 0.48,
        f"{ENV_ID}/heuristic4-v0": 0.43
    },
    (-1, f"{ENV_ID}/heuristic3-v0"): {
        f"{ENV_ID}/heuristic1-v0": 0.55,
        f"{ENV_ID}/heuristic2-v0": 0.40,
        f"{ENV_ID}/heuristic3-v0": 0.44,
        f"{ENV_ID}/heuristic4-v0": 0.26
    },
    (-1, f"{ENV_ID}/heuristic4-v0"): {
        f"{ENV_ID}/heuristic1-v0": 0.69,
        f"{ENV_ID}/heuristic2-v0": 0.53,
        f"{ENV_ID}/heuristic3-v0": 0.63,
        f"{ENV_ID}/heuristic4-v0": 0.48
    }
}

GREEDY_META_POLICY_MAP = meta_policy.get_greedy_policy_dict(PAIRWISE_RETURNS)
UNIFORM_META_POLICY_MAP = meta_policy.get_uniform_policy_dict(PAIRWISE_RETURNS)
SOFTMAX_META_POLICY_MAP = meta_policy.get_softmax_policy_dict(
    PAIRWISE_RETURNS, 0.25
)

BAPOSGMCP_KWARGS = {
    "discount": DISCOUNT,
    "c_init": 1.25,
    "c_base": 20000,
    "truncated": False,
    "action_selection": "pucb",
    "dirichlet_alpha": 0.6,
    "root_exploration_fraction": 0.5,
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": 50,
    "epsilon": 0.01
}


def get_baselines():   # noqa
    variable_params = {
        "num_sims": NUM_SIMS,
        "action_selection": ["pucb", "ucb", "uniform"]
    }

    baseline_params = []
    for (name, meta_policy_map) in [
            ("greedy", GREEDY_META_POLICY_MAP),
            ("softmax", SOFTMAX_META_POLICY_MAP),
            ("uniform", UNIFORM_META_POLICY_MAP)
    ]:
        baseline_params.extend(
            baseline_lib.load_all_baselines(
                variable_params=variable_params,
                baposgmcp_kwargs=BAPOSGMCP_KWARGS,
                policy_prior_map=POLICY_PRIOR_MAP,
                meta_policy_dict=GREEDY_META_POLICY_MAP,
                policy_id_suffix=name
            )
        )

    return baseline_params


def get_baposgmcps():   # noqa
    variable_params = {
        "num_sims": NUM_SIMS,
        "action_selection": ["pucb", "ucb", "uniform"]
    }

    baposgmcp_params = []
    for (name, meta_policy_map) in [
            ("greedy", GREEDY_META_POLICY_MAP),
            ("softmax", SOFTMAX_META_POLICY_MAP),
            ("uniform", UNIFORM_META_POLICY_MAP)
    ]:
        baposgmcp_params.extend(
            run_lib.load_baposgmcp_params(
                variable_params=variable_params,
                baposgmcp_kwargs=BAPOSGMCP_KWARGS,
                policy_prior_map=POLICY_PRIOR_MAP,
                meta_policy_dict=meta_policy_map,
                base_policy_id=f"baposgmcp_{name}"
            )
        )
    return baposgmcp_params


def get_fixed_baposgmcps():   # noqa
    variable_params = {
        "num_sims": NUM_SIMS,
        "action_selection": ["pucb", "ucb", "uniform"]
    }

    baposgmcp_params = baseline_lib.load_random_baposgmcp_params(
        variable_params=variable_params,
        baposgmcp_kwargs=BAPOSGMCP_KWARGS,
        policy_prior_map=POLICY_PRIOR_MAP,
        base_policy_id="baposgmcp_random"
    )
    baposgmcp_params.extend(
        baseline_lib.load_fixed_pi_baposgmcp_params(
            variable_params=variable_params,
            fixed_policy_ids=POLICY_IDS,
            baposgmcp_kwargs=BAPOSGMCP_KWARGS,
            policy_prior_map=POLICY_PRIOR_MAP,
            base_policy_id="baposgmcp_fixed"
        )
    )
    return baposgmcp_params



def main(args):   # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    other_params = run_lib.load_posggym_agent_params(POLICY_IDS)

    policy_params = get_baposgmcps()
    policy_params.extend(get_baselines())
    policy_params.extend(get_fixed_baposgmcps())

    exp_params_list = run_lib.get_pairwise_exp_params(
        ENV_ID,
        [policy_params, other_params],
        discount=DISCOUNT,
        exp_id_init=0,
        tracker_fn=run_lib.belief_tracker_fn,
        tracker_fn_kwargs={
            "num_agents": N_AGENTS,
            "step_limit": ENV_STEP_LIMIT,
            "discount": DISCOUNT
        },
        renderer_fn=None,
        **vars(args)
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
    exp_args["baposgmcp_kwargs"] = BAPOSGMCP_KWARGS

    print(f"== Using {args.n_procs} CPUs ==")
    run_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        using_ray=False,
        exp_args=vars(args),
        root_save_dir=args.root_save_dir,
        run_exp_id=args.run_exp_id
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = run_lib.get_exp_parser()
    parser.add_argument(
        "--get_num_exps", action="store_true",
        help="Compute and display number of experiments without running them."
    )
    main(parser.parse_args())

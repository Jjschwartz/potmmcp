"""Run BAPOSGMCP experiment in LBF env with heuristic policies.

This experiment compares performance using different Meta Policies.
"""
import copy
from pprint import pprint

import baposgmcp.run as run_lib
from baposgmcp import meta_policy
import baposgmcp.baselines as baseline_lib
from baposgmcp.run.render import EpisodeRenderer


ENV_NAME = "LBF5x5-n2-f3-static-v2"
DISCOUNT = 0.99
BAPOSGMCP_AGENT_ID = 0
OTHER_AGENT_ID = 1
POLICY_IDS = [
    f"{ENV_NAME}/heuristic1-v0",
    f"{ENV_NAME}/heuristic2-v0",
    f"{ENV_NAME}/heuristic3-v0",
    f"{ENV_NAME}/heuristic4-v0"
]
POLICY_PRIOR_MAP = {OTHER_AGENT_ID: {
    f"{ENV_NAME}/heuristic1-v0": 1/4,
    f"{ENV_NAME}/heuristic2-v0": 1/4,
    f"{ENV_NAME}/heuristic3-v0": 1/4,
    f"{ENV_NAME}/heuristic4-v0": 1/4
}}

PAIRWISE_RETURNS = {
    (-1, f"{ENV_NAME}/heuristic1-v0"): {
        f"{ENV_NAME}/heuristic1-v0": 0.48,
        f"{ENV_NAME}/heuristic2-v0": 0.28,
        f"{ENV_NAME}/heuristic3-v0": 0.34,
        f"{ENV_NAME}/heuristic4-v0": 0.28
    },
    (-1, f"{ENV_NAME}/heuristic2-v0"): {
        f"{ENV_NAME}/heuristic1-v0": 0.69,
        f"{ENV_NAME}/heuristic2-v0": 0.48,
        f"{ENV_NAME}/heuristic3-v0": 0.48,
        f"{ENV_NAME}/heuristic4-v0": 0.43
    },
    (-1, f"{ENV_NAME}/heuristic3-v0"): {
        f"{ENV_NAME}/heuristic1-v0": 0.55,
        f"{ENV_NAME}/heuristic2-v0": 0.40,
        f"{ENV_NAME}/heuristic3-v0": 0.44,
        f"{ENV_NAME}/heuristic4-v0": 0.26
    },
    (-1, f"{ENV_NAME}/heuristic4-v0"): {
        f"{ENV_NAME}/heuristic1-v0": 0.69,
        f"{ENV_NAME}/heuristic2-v0": 0.53,
        f"{ENV_NAME}/heuristic3-v0": 0.63,
        f"{ENV_NAME}/heuristic4-v0": 0.48
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


def get_baselines(args):   # noqa
    baseline_params = []
    for (name, meta_policy_map) in [
            ("greedy", GREEDY_META_POLICY_MAP),
            ("softmax", SOFTMAX_META_POLICY_MAP),
            ("uniform", UNIFORM_META_POLICY_MAP)
    ]:
        baseline_params.extend(baseline_lib.load_all_baselines(
            num_sims=args.num_sims,
            action_selection=['pucb'],
            baposgmcp_kwargs=BAPOSGMCP_KWARGS,
            other_policy_dist=POLICY_PRIOR_MAP,
            meta_policy_dict=meta_policy_map,
            policy_id_suffix=name
        ))
    return baseline_params


def get_baposgmcps(args):   # noqa
    baposgmcp_params = []
    for (name, meta_policy_map) in [
            ("greedy", GREEDY_META_POLICY_MAP),
            ("softmax", SOFTMAX_META_POLICY_MAP),
            ("uniform", UNIFORM_META_POLICY_MAP)
    ]:
        kwargs = copy.deepcopy(BAPOSGMCP_KWARGS)
        kwargs["policy_id"] = f"baposgmcp_{name}"
        baposgmcp_params.extend(run_lib.load_baposgmcp_params(
            num_sims=args.num_sims,
            baposgmcp_kwargs=kwargs,
            other_policy_dist=POLICY_PRIOR_MAP,
            meta_policy_dict=meta_policy_map
        ))
    return baposgmcp_params


def main(args):   # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    baposgmcp_params = get_baposgmcps(args)

    other_params = run_lib.load_posggym_agent_params(POLICY_IDS)

    exp_params_list = run_lib.get_baposgmcp_exp_params(
        ENV_NAME,
        baposgmcp_params,
        [other_params],
        discount=DISCOUNT,
        baposgmcp_agent_id=BAPOSGMCP_AGENT_ID,
        **vars(args)
    )

    if args.run_baselines:
        baseline_params = get_baselines(args)

        baseline_exp_params_list = run_lib.get_pairwise_exp_params(
            ENV_NAME,
            [baseline_params, other_params],
            discount=DISCOUNT,
            exp_id_init=exp_params_list[-1].exp_id+1,
            tracker_fn=None,
            tracker_fn_kwargs=None,
            renderer_fn=(lambda: [EpisodeRenderer()]) if args.render else None,
            **vars(args)
        )
        exp_params_list.extend(baseline_exp_params_list)

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_name = f"baposgmcp_heuristic_meta_pi_{seed_str}"

    exp_args = vars(args)
    exp_args["env_name"] = ENV_NAME
    exp_args["policy_ids"] = POLICY_IDS
    exp_args["policy_prior"] = POLICY_PRIOR_MAP
    exp_args["meta_policy"] = GREEDY_META_POLICY_MAP
    exp_args["discount"] = DISCOUNT
    exp_args["baposgmcp_kwargs"] = BAPOSGMCP_KWARGS

    if args.run_exp_id is not None:
        print(
            f"== Running Experiment {args.run_exp_id} of "
            f"{len(exp_params_list)} Experiments =="
        )
        exp_params_list = [exp_params_list[args.run_exp_id]]
    else:
        print(f"== Running {len(exp_params_list)} Experiments ==")

    print(f"== Using {args.n_procs} CPUs ==")
    run_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        using_ray=False,
        exp_args=vars(args),
        root_save_dir=args.root_save_dir
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = run_lib.get_exp_parser()
    parser.add_argument(
        "--init_seed", type=int, default=0,
        help="Experiment start seed."
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1,
        help="Number of seeds to use."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Experiment time limit, in seconds."
    )
    parser.add_argument(
        "--num_sims", type=int, nargs="*", default=[128],
        help="Number of simulations per search."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render experiment episodes."
    )
    parser.add_argument(
        "--record_env", action="store_true",
        help="Record renderings of experiment episodes."
    )
    parser.add_argument(
        "--render_tree", action="store_true",
        help="Render BAPOSGMCP search tree during experiment episodes."
    )
    parser.add_argument(
        "--run_baselines", action="store_true",
        help="Run baseline policies as well."
    )
    parser.add_argument(
        "--run_exp_id", type=int, default=None,
        help="Run only exp with specific ID. If None will run all exps."
    )
    main(parser.parse_args())

"""Run BAPOSGMCP experiment in LBF env with heuristic policies."""
import copy
from pprint import pprint

import baposgmcp.run as run_lib
import baposgmcp.baselines as baseline_lib
from baposgmcp.run.render import EpisodeRenderer

ENV_NAME = "LBF10x10-n2-f7-static-v2"
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
META_POLICY_MAP = {
    (-1, f"{ENV_NAME}/heuristic1-v0"): {
        f"{ENV_NAME}/heuristic1-v0": 1.0
    },
    (-1, f"{ENV_NAME}/heuristic2-v0"): {
        f"{ENV_NAME}/heuristic1-v0": 1.0
    },
    (-1, f"{ENV_NAME}/heuristic3-v0"): {
        f"{ENV_NAME}/heuristic1-v0": 1.0
    },
    (-1, f"{ENV_NAME}/heuristic4-v0"): {
        f"{ENV_NAME}/heuristic3-v0": 1.0
    }
}
BAPOSGMCP_KWARGS = {
    "discount": DISCOUNT,
    "c_init": 1.25,
    "c_base": 20000,
    "truncated": False,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": 50,
    "epsilon": 0.01
}


def get_baselines(args):   # noqa
    baseline_params = baseline_lib.load_all_baselines(
        num_sims=args.num_sims,
        action_selection=('pucb', 'ucb', 'uniform'),
        baposgmcp_kwargs=BAPOSGMCP_KWARGS,
        other_policy_dist=POLICY_PRIOR_MAP,
        meta_policy_dict=META_POLICY_MAP
    )

    # BAPOSGMCP using UCB action selection
    kwargs = copy.deepcopy(BAPOSGMCP_KWARGS)
    kwargs["action_selection"] = "ucb"
    kwargs["policy_id"] = "baposgmcp_ucb"
    baposgmcp_ucb_params = run_lib.load_baposgmcp_params(
        num_sims=args.num_sims,
        baposgmcp_kwargs=kwargs,
        other_policy_dist=POLICY_PRIOR_MAP,
        meta_policy_dict=META_POLICY_MAP
    )
    baseline_params.extend(baposgmcp_ucb_params)

    return baseline_params


def main(args):   # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    baposgmcp_params = run_lib.load_baposgmcp_params(
        num_sims=args.num_sims,
        baposgmcp_kwargs=BAPOSGMCP_KWARGS,
        other_policy_dist=POLICY_PRIOR_MAP,
        meta_policy_dict=META_POLICY_MAP
    )

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
    exp_name = f"baposgmcp_heuristic_{seed_str}"

    exp_args = vars(args)
    exp_args["env_name"] = ENV_NAME
    exp_args["policy_ids"] = POLICY_IDS
    exp_args["policy_prior"] = POLICY_PRIOR_MAP
    exp_args["meta_policy"] = META_POLICY_MAP
    exp_args["discount"] = DISCOUNT
    exp_args["baposgmcp_kwargs"] = BAPOSGMCP_KWARGS

    if args.run_exp_id is not None:
        exp_params_list = [exp_params_list[args.run_exp_id]]

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
        "--run_baselines", action="store_true",
        help="Run baseline policies as well."
    )
    parser.add_argument(
        "--run_exp_id", type=int, default=None,
        help="Run only exp with specific ID. If None will run all exps."
    )
    main(parser.parse_args())

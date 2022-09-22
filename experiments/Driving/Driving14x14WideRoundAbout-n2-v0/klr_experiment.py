"""Run BAPOSGMCP experiment in Driving env with KLR policies."""
from typing import List
from pprint import pprint

import baposgmcp.run as run_lib
import baposgmcp.baselines as baseline_lib
from baposgmcp.run.render import EpisodeRenderer


ENV_NAME = "Driving14x14WideRoundAbout-n2-v0"
DISCOUNT = 0.99
BAPOSGMCP_AGENT_ID = 0
OTHER_AGENT_ID = 1
MIN_K = 0
MAX_K = 4
BAPOSGMCP_KWARGS = {
    "discount": DISCOUNT,
    "c_init": 1.25,
    "c_base": 20000,
    "truncated": True,
    "action_selection": "pucb",
    "dirichlet_alpha": 0.5,    # 5 actions / 10
    "root_exploration_fraction": 0.5,   # half actions valid/useful at any step
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": 50,
    "epsilon": 0.01
}

def get_policy_ids(seed: int, min_k: int = 0, max_k: int = 4) -> List[str]:  # noqa
    return [
        f"{ENV_NAME}/klr_k{k}_seed{seed}-v0" for k in range(min_k, max_k+1)
    ]


def get_policy_prior(seed: int, max_k: int):   # noqa
    policy_ids = get_policy_ids(seed, min_k=MIN_K, max_k=max_k-1)
    return {OTHER_AGENT_ID: {
        pi_id: 1.0 / len(policy_ids) for pi_id in policy_ids
    }}


def get_meta_policy(seed: int, max_k: int):   # noqa
    return {
        (-1, f"{ENV_NAME}/klr_k{k}_seed{seed}-v0",): {
            f"{ENV_NAME}/klr_k{k+1}_seed{seed}-v0": 1.0
        }
        for k in range(MIN_K, max_k)
    }


def get_baselines(args, policy_prior, meta_policy_dict):   # noqa
    baseline_params = baseline_lib.load_all_baselines(
        num_sims=args.num_sims,
        action_selection=['pucb'],
        baposgmcp_kwargs=BAPOSGMCP_KWARGS,
        other_policy_dist=policy_prior,
        meta_policy_dict=meta_policy_dict
    )
    return baseline_params


def main(args):   # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    policy_prior = get_policy_prior(args.pop_seed, args.k)
    meta_policy_dict = get_meta_policy(args.pop_seed, args.k)
    baposgmcp_params = run_lib.load_baposgmcp_params(
        num_sims=args.num_sims,
        baposgmcp_kwargs=BAPOSGMCP_KWARGS,
        other_policy_dist=policy_prior,
        meta_policy_dict=meta_policy_dict
    )

    other_policy_ids = get_policy_ids(
        args.pop_seed, min_k=MIN_K, max_k=args.k
    )
    other_params = run_lib.load_posggym_agent_params(other_policy_ids)

    exp_params_list = run_lib.get_baposgmcp_exp_params(
        ENV_NAME,
        baposgmcp_params,
        [other_params],
        discount=DISCOUNT,
        baposgmcp_agent_id=BAPOSGMCP_AGENT_ID,
        **vars(args)
    )

    if args.run_baselines:
        baseline_params = get_baselines(args, policy_prior, meta_policy_dict)

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
    exp_str = "" if args.run_exp_id is None else f"_exp{args.run_exp_id}"
    exp_name = f"baposgmcp_klr{exp_str}_{seed_str}"

    exp_args = vars(args)
    exp_args["env_name"] = ENV_NAME
    exp_args["discount"] = DISCOUNT
    exp_args["policy_prior"] = policy_prior
    exp_args["meta_policy"] = meta_policy_dict
    exp_args["baposgmcp_kwargs"] = BAPOSGMCP_KWARGS

    if args.run_exp_id is not None:
        print(
            f"== Running Experiment {args.run_exp_id} of "
            f"{len(exp_params_list)} Experiments =="
        )
        exp_params_list = [exp_params_list[args.run_exp_id]]
    else:
        print(f"== Running {len(exp_params_list)} Experiments ==")

    print(f"== Running {len(exp_params_list)} Experiments ==")
    print(f"== Using {args.n_procs} CPUs ==")
    run_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        using_ray=True,
        exp_args=vars(args),
        root_save_dir=args.root_save_dir
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = run_lib.get_exp_parser()
    parser.add_argument(
        "--k", type=int, default=1,
        help="Max reasoning level K in [1, 4]."
    )
    parser.add_argument(
        "--pop_seed", type=int, default=0,
        help="Population seed of policies to use [0, 4]."
    )
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

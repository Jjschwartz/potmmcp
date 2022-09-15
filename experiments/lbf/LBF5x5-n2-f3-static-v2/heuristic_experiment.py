"""Run BAPOSGMCP experiment in LBF env with heuristic policies."""
from pprint import pprint

import posggym_agents

from baposgmcp.baselines.po_meta import POMeta
from baposgmcp.run.render import EpisodeRenderer
from baposgmcp.baselines.meta import MetaBaselinePolicy
from baposgmcp.run.tree_exp import load_baposgmcp_params
from baposgmcp.run.tree_exp import get_baposgmcp_exp_params
from baposgmcp.run.exp import (
    run_experiments, PolicyParams, get_exp_parser, get_pairwise_exp_params
)

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
        f"{ENV_NAME}/heuristic1-v0": 1.0
    }
}
BAPOSGMCP_KWARGS = {
    "c_init": 1.25,
    "c_base": 20000,
    "truncated": False,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": 50,
    "epsilon": 0.01
}


def get_entry_point(policy_id: str):   # noqa

    def entry_point(model, agent_id, kwargs):
        return posggym_agents.make(policy_id, model, agent_id, **kwargs)

    return entry_point


def main(args):   # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    baposgmcp_params = load_baposgmcp_params(
        ENV_NAME,
        agent_id=0,
        discount=DISCOUNT,
        num_sims=args.num_sims,
        baposgmcp_kwargs=BAPOSGMCP_KWARGS,
        other_policy_dist=POLICY_PRIOR_MAP,
        meta_policy_dict=META_POLICY_MAP
    )

    other_params = [
        PolicyParams(
            id=policy_id,
            entry_point=get_entry_point(policy_id),
            kwargs={},
            info=None
        )
        for policy_id in POLICY_IDS
    ]

    exp_params_list = get_baposgmcp_exp_params(
        ENV_NAME,
        baposgmcp_params,
        [other_params],
        discount=DISCOUNT,
        baposgmcp_agent_id=BAPOSGMCP_AGENT_ID,
        **vars(args)
    )
    # TODO remove
    exp_params_list = []

    if args.run_baselines:
        baseline_params = [
            # PolicyParams(
            #     id="metabaseline",
            #     entry_point=MetaBaselinePolicy.posggym_agents_entry_point,
            #     kwargs={
            #         "other_policy_dist": POLICY_PRIOR_MAP,
            #         "meta_policy_dict": META_POLICY_MAP
            #     }
            # ),
            PolicyParams(
                id="POMeta",
                entry_point=POMeta.posggym_agents_entry_point,
                kwargs={
                    "belief_size": 1000,
                    "other_policy_dist": POLICY_PRIOR_MAP,
                    "meta_policy_dict": META_POLICY_MAP,
                }
            )
        ]
        baseline_exp_params_list = get_pairwise_exp_params(
            ENV_NAME,
            [baseline_params, other_params],
            discount=DISCOUNT,
            # TODO change this
            # exp_id_init=exp_params_list[-1].exp_id,
            exp_id_init=0,
            tracker_fn=None,
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

    print(f"== Running {len(exp_params_list)} Experiments ==")
    print(f"== Using {args.n_procs} CPUs ==")
    run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        using_ray=False,
        exp_args=vars(args)
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = get_exp_parser()
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
    main(parser.parse_args())

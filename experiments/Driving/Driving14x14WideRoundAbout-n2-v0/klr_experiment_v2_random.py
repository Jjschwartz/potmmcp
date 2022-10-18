"""Run BAPOSGMCP experiment in Driving env with KLR policies.

V2 - running only relevant exps after V1

Namely,
- using only PUCB
- using truncated search

Only Greedy meta-policy for PO-META

"""
import copy
from pprint import pprint

import baposgmcp.run as run_lib
from baposgmcp import meta_policy
import baposgmcp.baselines as baseline_lib


ENV_ID = "Driving14x14WideRoundAbout-n2-v0"
N_AGENTS = 2
ENV_STEP_LIMIT = 50
POLICY_SEED = 0

DISCOUNT = 0.99
NUM_SIMS = [10, 50, 100, 500, 1000]
BAPOSGMCP_AGENT_ID = 0
OTHER_AGENT_ID = 1

# NOTE There are 5 policies available (K=4) but we assume the other agent is
# only using 4 (K=3), but we have all (K=4) policies available for the
# meta policy
POLICY_IDS = [
    f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0",
    f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0",
    f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0",
    f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0",
    f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0"
]
# Defined for K=3
POLICY_PRIOR_MAP = {OTHER_AGENT_ID: {
    f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 1/4,
    f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 1/4,
    f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 1/4,
    f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 1/4
}}
# Defined for policy states with (K=3) and meta-policy (K=4)
PAIRWISE_RETURNS = {
    (-1, f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0"): {
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 1.91,
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 2.21,
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 2.01,
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 1.74,
        f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0": 1.70
    },
    (-1, f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0"): {
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 2.21,
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 2.05,
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 2.21,
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 1.74,
        f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0": 1.88
    },
    (-1, f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0"): {
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 1.97,
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 2.19,
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 2.12,
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 2.18,
        f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0": 1.76
    },
    (-1, f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0"): {
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 1.73,
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 1.76,
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 2.18,
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 1.89,
        f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0": 2.21
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
    "truncated": True,
    "action_selection": "pucb",
    "dirichlet_alpha": 0.5,    # 5 actions / 10
    "root_exploration_fraction": 0.5,   # half actions valid/useful at any step
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": ENV_STEP_LIMIT,
    "epsilon": 0.01
}


def get_fixed_baposgmcps():   # noqa
    variable_params = {
        "num_sims": NUM_SIMS,
    }

    random_kwargs = copy.deepcopy(BAPOSGMCP_KWARGS)
    random_kwargs["truncated"] = False
    baposgmcp_params = baseline_lib.load_random_baposgmcp_params(
        variable_params=variable_params,
        baposgmcp_kwargs=random_kwargs,
        policy_prior_map=POLICY_PRIOR_MAP,
        base_policy_id="baposgmcp_random"
    )
    # NUM Exps:
    # = |NUM_SIMS|
    # = 5
    assert len(baposgmcp_params) == len(NUM_SIMS)
    return baposgmcp_params


def main(args):   # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    other_params = run_lib.load_posggym_agent_params(
        list(POLICY_PRIOR_MAP[OTHER_AGENT_ID])
    )
    assert len(other_params) == len(POLICY_PRIOR_MAP[OTHER_AGENT_ID])

    policy_params = get_fixed_baposgmcps()

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
    exp_name = f"baposgmcp_klr{exp_str}_{seed_str}"

    exp_args = vars(args)
    exp_args["env_id"] = ENV_ID
    exp_args["discount"] = DISCOUNT
    exp_args["baposgmcp_kwargs"] = BAPOSGMCP_KWARGS

    print(f"== Using {args.n_procs} CPUs ==")
    run_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        using_ray=True,
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

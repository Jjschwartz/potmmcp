"""Run BAPOSGMCP experiment in PP env with teams of SP agents."""
import copy
from pprint import pprint

import baposgmcp.run as run_lib
from baposgmcp import meta_policy
import baposgmcp.baselines as baseline_lib

ENV_ID = "PredatorPrey10x10-P2-p3-s2-coop-v0"
N_AGENTS = 2
ENV_STEP_LIMIT = 50

DISCOUNT = 0.99
NUM_SIMS = [10, 50, 100, 500, 1000]
BAPOSGMCP_AGENT_ID = 0
OTHER_AGENT_IDS = 1

POLICY_IDS = [
    f"{ENV_ID}/sp_seed0-v0",
    f"{ENV_ID}/sp_seed1-v0",
    f"{ENV_ID}/sp_seed2-v0",
    f"{ENV_ID}/sp_seed3-v0",
    f"{ENV_ID}/sp_seed4-v0"
]
# Defined for K=3
POLICY_PRIOR_MAP = {
    (-1, 'sp_seed0-v0'): 0.2,
    (-1, 'sp_seed1-v0'): 0.2,
    (-1, 'sp_seed2-v0'): 0.2,
    (-1, 'sp_seed3-v0'): 0.2,
    (-1, 'sp_seed4-v0'): 0.2,
}
# Add env_id to each policy id
pi_states = list(POLICY_PRIOR_MAP)
for pi_state in pi_states:
    updated_pi_state = tuple(
        v if v == -1 else f"{ENV_ID}/{v}" for v in pi_state
    )
    prob = POLICY_PRIOR_MAP.pop(pi_state)
    POLICY_PRIOR_MAP[updated_pi_state] = prob

# Defined for policy states with (K=3) and meta-policy (K=4)
PAIRWISE_RETURNS = {
    (-1, "sp_seed0-v0"): {
        "sp_seed0-v0": 0.98,
        "sp_seed1-v0": 0.29,
        "sp_seed2-v0": 0.47,
        "sp_seed3-v0": 0.32,
        "sp_seed4-v0": 0.64,
    },
    (-1, "sp_seed1-v0"): {
        "sp_seed0-v0": 0.29,
        "sp_seed1-v0": 0.98,
        "sp_seed2-v0": 0.31,
        "sp_seed3-v0": 0.30,
        "sp_seed4-v0": 0.56,
    },
    (-1, "sp_seed2-v0"): {
        "sp_seed0-v0": 0.47,
        "sp_seed1-v0": 0.31,
        "sp_seed2-v0": 0.99,
        "sp_seed3-v0": 0.71,
        "sp_seed4-v0": 0.37,
    },
    (-1, "sp_seed3-v0"): {
        "sp_seed0-v0": 0.32,
        "sp_seed1-v0": 0.30,
        "sp_seed2-v0": 0.71,
        "sp_seed3-v0": 0.99,
        "sp_seed4-v0": 0.38,
    },
    (-1, "sp_seed4-v0"): {
        "sp_seed0-v0": 0.64,
        "sp_seed1-v0": 0.56,
        "sp_seed2-v0": 0.37,
        "sp_seed3-v0": 0.38,
        "sp_seed4-v0": 0.99,
    },
}
# Add env_id to each policy id
pi_states = list(PAIRWISE_RETURNS)
for pi_state in pi_states:
    updated_pairwise_returns = {
        f"{ENV_ID}/{k}": v for k, v in PAIRWISE_RETURNS[pi_state].items()
    }
    updated_pi_state = tuple(
        v if v == -1 else f"{ENV_ID}/{v}" for v in pi_state
    )
    PAIRWISE_RETURNS.pop(pi_state)
    PAIRWISE_RETURNS[updated_pi_state] = updated_pairwise_returns


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
    # = |NUM_SIMS| * (|PIS| + 1)
    # = 5 * (5 + 1)
    # = 30
    print(f"Num BAPOSGMCP-fixed params = {len(baposgmcp_params)}")
    assert len(baposgmcp_params) == len(NUM_SIMS)
    return baposgmcp_params


def main(args):   # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    other_params = run_lib.load_posggym_agent_params(POLICY_IDS)
    print(f"Num Other params = {len(other_params)}")

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

"""Run BAPOSGMCP experiment in Driving env with KLR policies."""
import math
from pprint import pprint

import baposgmcp.run as run_lib
from baposgmcp import meta_policy
import baposgmcp.baselines as baseline_lib


ENV_ID = "PursuitEvasion16x16-v0"
N_AGENTS = 2
ENV_STEP_LIMIT = 100
POLICY_SEED = 0

DISCOUNT = 0.99
NUM_SIMS = [10, 50, 100, 500, 1000]
BAPOSGMCP_AGENT_ID = 0
OTHER_AGENT_ID = 1

# NOTE There are 5 policies available (K=4) but we assume the other agent is
# only using 4 (K=3), but we have all (K=4) policies available for the
# meta policy
POLICY_IDS = {
    0: [
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0",
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0",
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0",
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0",
        f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0"
    ],
    1: [
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0",
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0",
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0",
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0",
        f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0"
    ],

}
# Defined for K=3
POLICY_PRIOR_MAP = {
    0: {
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": 1/4,
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": 1/4,
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": 1/4,
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": 1/4
    },
    1: {
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0": 1/4,
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0": 1/4,
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0": 1/4,
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0": 1/4
    }
}
# Gotta wrap it as algs expect Dict[ID, prior]
POLICY_PRIOR_MAP = {
    0: {0: POLICY_PRIOR_MAP[0]},
    1: {1: POLICY_PRIOR_MAP[1]}
}
# Defined for policy states with (K=3) and meta-policy (K=4)
PAIRWISE_RETURNS = {
    0: {
        (-1, f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": 0.21,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": 0.97,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": -0.16,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": -0.27,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": 0.60,
        },
        (-1, f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": -0.66,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": -0.25,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": 0.91,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": 0.02,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": -0.41,
        },
        (-1, f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": -0.60,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": -0.74,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": 0.25,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": 0.61,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": -0.20,
        },
        (-1, f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": 0.19,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": 0.59,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": -0.78,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": -0.45,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": 0.97,
        },
        (-1, f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": 0.17,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": 0.67,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": -0.36,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": -0.78,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": 0.68,
        }
    },
    1: {
        (f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0", -1): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0": -0.21,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0": 0.66,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0": 0.60,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0": -0.19,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0": -0.17,
        },
        (f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0", -1): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0": -0.97,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0": 0.25,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0": 0.74,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0": -0.59,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0": -0.67,
        },
        (f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0", -1): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0": 0.16,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0": -0.91,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0": -0.25,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0": 0.78,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0": 0.36,
        },
        (f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0", -1): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0": 0.27,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0": -0.02,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0": -0.61,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0": 0.45,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0": 0.78,
        },
        (f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0", -1): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0": -0.60,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0": 0.41,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0": 0.20,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0": -0.97,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0": -0.68,
        },
    }
}

GREEDY_META_POLICY_MAP = {
    0: meta_policy.get_greedy_policy_dict(PAIRWISE_RETURNS[0]),
    1: meta_policy.get_greedy_policy_dict(PAIRWISE_RETURNS[1])
}
UNIFORM_META_POLICY_MAP = {
    0: meta_policy.get_uniform_policy_dict(PAIRWISE_RETURNS[0]),
    1: meta_policy.get_uniform_policy_dict(PAIRWISE_RETURNS[1])
}
SOFTMAX_META_POLICY_MAP = {
    0: meta_policy.get_softmax_policy_dict(PAIRWISE_RETURNS[0], 0.25),
    1: meta_policy.get_softmax_policy_dict(PAIRWISE_RETURNS[1], 0.25)
}


BAPOSGMCP_KWARGS = {
    "discount": DISCOUNT,
    "c_init": 1.25,
    "c_base": 20000,
    # "truncated": True,    # to be set
    # "action_selection": "pucb",   # to be set
    "dirichlet_alpha": 0.4,    # 4 actions / 10
    "root_exploration_fraction": 0.5,   # half actions valid/useful at any step
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": ENV_STEP_LIMIT,
    "epsilon": 0.01
}


def get_baselines(agent_id: int, other_agent_id: int):   # noqa
    variable_params = {
        "num_sims": NUM_SIMS,
        # "action_selection": ["pucb", "ucb", "uniform"],
        # "truncated": [True, False]
        "action_selection": ["pucb"],
        "truncated": [True]
    }

    meta_pis = [
        ("greedy", GREEDY_META_POLICY_MAP[agent_id]),
        # ("softmax", SOFTMAX_META_POLICY_MAP),
        # ("uniform", UNIFORM_META_POLICY_MAP)
    ]

    baseline_params = []
    for (name, meta_policy_map) in meta_pis:
        baseline_params.extend(
            baseline_lib.load_all_baselines(
                variable_params=variable_params,
                baposgmcp_kwargs=BAPOSGMCP_KWARGS,
                policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
                meta_policy_dict=meta_policy_map,
                policy_id_suffix=f"{name}_i{agent_id}"
            )
        )
    # NUM Exps:
    # = |metabaseline| + |POMeta| + |POMetaRollout|
    # = (|Meta|) + (|NUM_SIMS| * |Meta|)
    #   + (|NUM_SIMS| * |ACT SEL| * |Truncated| * |Meta|)

    # with truncated and different action selections
    # = (3) + (5 * 3) + (5 * 3 * 2 * 3)
    # = 3 + 15 + 90
    # = 108
    # assert (
    #   len(baseline_params) == (3 + (len(NUM_SIMS)*3) + (len(NUM_SIMS)*3*2*3))
    # )

    # without truncated options and only pucb and greedy meta-pi
    # = (1) + (5 * 1) + (5 * 1 * 1 * 1)
    # = 11
    exp_num_params = (
        len(meta_pis)
        + len(NUM_SIMS)*len(meta_pis)
        + math.prod(len(v) for v in variable_params.values())*len(meta_pis)
    )
    assert len(baseline_params) == exp_num_params
    print(f"Num Baseline i{agent_id} params = {exp_num_params}")
    return baseline_params


def get_baposgmcps(agent_id: int, other_agent_id: int):   # noqa
    variable_params = {
        "num_sims": NUM_SIMS,
        # "action_selection": ["pucb", "ucb", "uniform"],
        # "truncated": [True, False]
        "action_selection": ["pucb"],
        "truncated": [True]
    }

    meta_pis = [
        ("greedy", GREEDY_META_POLICY_MAP[agent_id]),
        ("softmax", SOFTMAX_META_POLICY_MAP[agent_id]),
        ("uniform", UNIFORM_META_POLICY_MAP[agent_id])
    ]

    baposgmcp_params = []
    for (name, meta_policy_map) in meta_pis:
        baposgmcp_params.extend(
            run_lib.load_baposgmcp_params(
                variable_params=variable_params,
                baposgmcp_kwargs=BAPOSGMCP_KWARGS,
                policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
                meta_policy_dict=meta_policy_map,
                base_policy_id=f"baposgmcp_{name}_i{agent_id}"
            )
        )
    # NUM Exps:
    # = |NUM_SIMS| * |ACT SEL| * |Truncated| * |Meta|

    # With truncated/not and all act selection options
    # = 5 * 3 * 2 * 3
    # = 90
    # assert len(baposgmcp_params) == (len(NUM_SIMS)*3*2*3)

    # With truncated and PUCB
    # = 5 * 1 * 1 * 3
    # = 15
    exp_num_params = (
        math.prod(len(v) for v in variable_params.values()) * len(meta_pis)
    )
    assert len(baposgmcp_params) == exp_num_params
    print(f"Num BAPOSGMCP i{agent_id} params = {exp_num_params}")
    return baposgmcp_params


def get_fixed_baposgmcps(agent_id: int, other_agent_id: int):   # noqa
    # random only works for 'truncated=False'
    random_variable_params = {
        "num_sims": NUM_SIMS,
        # "action_selection": ["pucb", "ucb", "uniform"],
        "action_selection": ["pucb"],
        "truncated": [False]
    }
    baposgmcp_params = baseline_lib.load_random_baposgmcp_params(
        variable_params=random_variable_params,
        baposgmcp_kwargs=BAPOSGMCP_KWARGS,
        policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
        base_policy_id=f"baposgmcp-random_i{agent_id}"
    )

    fixed_variable_params = {
        "num_sims": NUM_SIMS,
        # "action_selection": ["pucb", "ucb", "uniform"],
        # "truncated": [True, False]
        "action_selection": ["pucb"],
        "truncated": [True]
    }
    baposgmcp_params.extend(
        baseline_lib.load_fixed_pi_baposgmcp_params(
            variable_params=fixed_variable_params,
            fixed_policy_ids=POLICY_IDS[agent_id],
            baposgmcp_kwargs=BAPOSGMCP_KWARGS,
            policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
            base_policy_id=f"baposgmcp-fixed_i{agent_id}"
        )
    )
    # NUM Exps:
    # = (|NUM_SIMS|*|ACT SEL|) + (|NUM_SIMS|*|ACT SEL|*|Truncated|*|PIS|)

    # With truncated/not and all act selection options
    # = (5 * 3) + (5 * 3 * 2 * 5)
    # = 165

    # With truncated and PUCB
    # = (5 * 1) + (5 * 1 * 1 * 5)
    # = 30
    exp_num_params = (
        math.prod(len(v) for v in random_variable_params.values())
        + (
            math.prod(len(v) for v in fixed_variable_params.values())
            * (len(POLICY_IDS[agent_id]))
        )
    )
    assert len(baposgmcp_params) == exp_num_params
    print(f"Num fixed/random BAPOSGMCP i{agent_id} params = {exp_num_params}")
    return baposgmcp_params


def main(args):   # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    exp_params_list = []
    for (i, j) in [(0, 1), (1, 0)]:
        other_params = run_lib.load_posggym_agent_params(
            list(POLICY_PRIOR_MAP[j][j])
        )
        print(f"Num Other agent params j{j} = {len(other_params)}")
        assert len(other_params) == len(POLICY_PRIOR_MAP[j][j])

        policy_params = get_baposgmcps(i, j)
        policy_params.extend(get_baselines(i, j))
        policy_params.extend(get_fixed_baposgmcps(i, j))

        if i == 0:
            agent_0_params = policy_params
            agent_1_params = other_params
        else:
            agent_0_params = other_params
            agent_1_params = policy_params

        exp_params_list.extend(run_lib.get_pairwise_exp_params(
            ENV_ID,
            [agent_0_params, agent_1_params],
            discount=DISCOUNT,
            exp_id_init=len(exp_params_list),
            tracker_fn=run_lib.belief_tracker_fn,
            tracker_fn_kwargs={
                "num_agents": N_AGENTS,
                "step_limit": ENV_STEP_LIMIT,
                "discount": DISCOUNT
            },
            renderer_fn=None,
            **vars(args)
        ))

    if args.get_num_exps:
        print(f"Number of experiments = {len(exp_params_list)}")
        return

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_str = "" if args.run_exp_id is None else f"_exp{args.run_exp_id}"
    exp_name = f"klr{exp_str}_{seed_str}"

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

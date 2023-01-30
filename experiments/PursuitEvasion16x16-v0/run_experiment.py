"""Run BAPOSGMCP experiment in PursuitEvasion env with KLR policies."""
import copy
import math
from pprint import pprint
from typing import Any, Dict, List

import baposgmcp.baselines as baseline_lib
import baposgmcp.run as run_lib
from baposgmcp import meta_policy


ENV_ID = "PursuitEvasion16x16-v0"
N_AGENTS = 2
ENV_STEP_LIMIT = 100
POLICY_SEED = 0

DISCOUNT = 0.99
NUM_SIMS = [10, 50, 100, 500, 1000]
BAPOSGMCP_AGENT_ID = 0
OTHER_AGENT_ID = 1

# DEBUG - Delete/comment this
# NUM_SIMS = [2, 5]

# NOTE There are 5 policies available (K=4) but we assume the other agent is
# only using 4 (K=3), but we have all (K=4) policies available for the
# meta policy
POLICY_IDS = {
    0: [
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0",
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0",
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0",
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0",
        f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0",
    ],
    1: [
        f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0",
        f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0",
        f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0",
        f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0",
        f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0",
    ],
}
# Defined for K=3
# Gotta wrap it as algs expect Dict[ID, prior]
POLICY_PRIOR_MAP = {
    0: {
        0: {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": 1 / 4,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": 1 / 4,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": 1 / 4,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": 1 / 4,
        },
    },
    1: {
        1: {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0": 1 / 4,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0": 1 / 4,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0": 1 / 4,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0": 1 / 4,
        },
    }
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
        },
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
    },
}

GREEDY_META_POLICY_MAP = {
    0: meta_policy.get_greedy_policy_dict(PAIRWISE_RETURNS[0]),
    1: meta_policy.get_greedy_policy_dict(PAIRWISE_RETURNS[1]),
}
UNIFORM_META_POLICY_MAP = {
    0: meta_policy.get_uniform_policy_dict(PAIRWISE_RETURNS[0]),
    1: meta_policy.get_uniform_policy_dict(PAIRWISE_RETURNS[1]),
}
SOFTMAX_META_POLICY_MAP = {
    0: meta_policy.get_softmax_policy_dict(PAIRWISE_RETURNS[0], 0.25),
    1: meta_policy.get_softmax_policy_dict(PAIRWISE_RETURNS[1], 0.25),
}

BEST_META_PI_NAME = "softmax"
BEST_META_PI_MAP = SOFTMAX_META_POLICY_MAP


BAPOSGMCP_PUCT_KWARGS = {
    "discount": DISCOUNT,
    "c": 1.25,
    # "truncated": True,   # added as variable param like num sims
    "action_selection": "pucb",
    "dirichlet_alpha": 0.4,  # 4 actions / 10
    "root_exploration_fraction": 0.5,  # half actions valid/useful at any step
    "reinvigorator": None,  # Use default rejection sampler
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    "step_limit": ENV_STEP_LIMIT,
    "epsilon": 0.01,
}
UCB_C = math.sqrt(2.0)  # as per OG paper/standard parameter


def get_baselines(agent_id: int, other_agent_id: int):  # noqa
    baseline_params = []
    for (name, meta_policy_map) in [
        ("greedy", GREEDY_META_POLICY_MAP[agent_id]),
        ("softmax", SOFTMAX_META_POLICY_MAP[agent_id]),
        ("uniform", UNIFORM_META_POLICY_MAP[agent_id]),
    ]:
        # Meta Baseline Policy
        policy_id = f"metabaseline_{name}_i{agent_id}"
        policy_params = run_lib.PolicyParams(
            id=policy_id,
            entry_point=baseline_lib.MetaBaselinePolicy.posggym_agents_entry_point,
            kwargs={
                "policy_id": policy_id,
                "policy_prior_map": POLICY_PRIOR_MAP[other_agent_id],
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


def get_baposgmcps(agent_id: int, other_agent_id: int):  # noqa
    variable_params: Dict[str, List[Any]] = {"num_sims": NUM_SIMS, "truncated": [True]}

    meta_pis = [
        ("greedy", GREEDY_META_POLICY_MAP[agent_id]),
        ("softmax", SOFTMAX_META_POLICY_MAP[agent_id]),
        ("uniform", UNIFORM_META_POLICY_MAP[agent_id]),
    ]

    baposgmcp_params = []
    for (name, meta_policy_map) in meta_pis:
        baposgmcp_params.extend(
            run_lib.load_baposgmcp_params(
                variable_params=variable_params,
                baposgmcp_kwargs=BAPOSGMCP_PUCT_KWARGS,
                policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
                meta_policy_dict=meta_policy_map,
                base_policy_id=f"baposgmcp_meta{name}_i{agent_id}",
            )
        )
    # NUM Exps:
    # = |NUM_SIMS| * |Meta|
    # = 5 * 3
    # = 15
    assert len(baposgmcp_params) == (len(NUM_SIMS) * 3)
    return baposgmcp_params


def get_fixed_baposgmcps(agent_id: int, other_agent_id: int):  # noqa
    random_kwargs = copy.deepcopy(BAPOSGMCP_PUCT_KWARGS)
    random_kwargs["truncated"] = False
    baposgmcp_params = baseline_lib.load_random_baposgmcp_params(
        variable_params={
            "num_sims": NUM_SIMS,
            "truncated": [False],  # added so it's clearly visible in policy id
        },
        baposgmcp_kwargs=random_kwargs,
        policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
        base_policy_id=f"baposgmcp-random_i{agent_id}",
    )

    baposgmcp_params.extend(
        baseline_lib.load_fixed_pi_baposgmcp_params(
            variable_params={
                "num_sims": NUM_SIMS,
                "truncated": [True],
            },
            fixed_policy_ids=POLICY_IDS[agent_id],
            baposgmcp_kwargs=BAPOSGMCP_PUCT_KWARGS,
            policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
            base_policy_id=f"baposgmcp-fixed_i{agent_id}",
        )
    )

    # NUM Exps:
    # = |NUM_SIMS| * (|PIS| + 1)
    # = 5 * (5 + 1)
    # = 30
    assert len(baposgmcp_params) == (len(NUM_SIMS) * (len(POLICY_IDS[agent_id]) + 1))
    return baposgmcp_params


def get_ucb_mcps(agent_id: int, other_agent_id: int):  # noqa
    """
    - UCB MCP (UCB + Meta-Policy)
      - Best meta-policy [Truncated]
    - UCB MCP Random (UCB + Random) [Untruncated]
    """
    ucb_kwargs = copy.deepcopy(BAPOSGMCP_PUCT_KWARGS)
    ucb_kwargs["c"] = UCB_C
    ucb_kwargs["action_selection"] = "ucb"

    ucb_params = run_lib.load_baposgmcp_params(
        variable_params={"num_sims": NUM_SIMS, "truncated": [True]},
        baposgmcp_kwargs=ucb_kwargs,
        policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
        meta_policy_dict=BEST_META_PI_MAP[agent_id],
        base_policy_id=f"ucbmcp_meta{BEST_META_PI_NAME}_i{agent_id}",
    )

    random_kwargs = copy.deepcopy(ucb_kwargs)
    random_kwargs["truncated"] = False
    ucb_params.extend(
        baseline_lib.load_random_baposgmcp_params(
            variable_params={"num_sims": NUM_SIMS, "truncated": [False]},
            baposgmcp_kwargs=random_kwargs,
            policy_prior_map=POLICY_PRIOR_MAP[other_agent_id],
            base_policy_id=f"ucbmcp-random_i{agent_id}",
        )
    )

    # NUM Exps:
    # = |NUM_SIMS| * 1 + |NUM_SIMS|
    # = 5 * 1 + 5
    # = 10
    assert len(ucb_params) == len(NUM_SIMS) * 2
    return ucb_params


def main(args):  # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    exp_params_list = []
    for (i, j) in [(0, 1), (1, 0)]:
        other_params = run_lib.load_posggym_agent_params(list(POLICY_PRIOR_MAP[j][j]))
        print(f"Num Other agent params j{j} = {len(other_params)}")
        assert len(other_params) == len(POLICY_PRIOR_MAP[j][j])

        policy_params = get_baposgmcps(i, j)
        # - BAPOSGMCP (PUCB + Meta-Policy)
        #   - Greedy [Truncated]
        #   - Softmax [Truncated]
        #   - Uniform [Truncated]
        policy_params = get_baposgmcps(i, j)
        # - BAPOSGMCP (PUCB + Random) [Untruncated]
        # - BAPOSGMCP (PUCB + Fixed policies) [Truncated]
        policy_params.extend(get_fixed_baposgmcps(i, j))
        # - UCB MCP (UCB + Meta-Policy)
        #   - Best meta-policy [Truncated]
        # - UCB MCP Random (UCB + Random) [Untruncated]
        policy_params.extend(get_ucb_mcps(i, j))
        # - Meta-Policy
        #   - Greedy
        #   - Softmax
        #   - Uniform
        policy_params.extend(get_baselines(i, j))

        if i == 0:
            agent_0_params = policy_params
            agent_1_params = other_params
        else:
            agent_0_params = other_params
            agent_1_params = policy_params

        exp_params_list.extend(
            run_lib.get_pairwise_exp_params(
                ENV_ID,
                [agent_0_params, agent_1_params],
                discount=DISCOUNT,
                exp_id_init=len(exp_params_list),
                tracker_fn=run_lib.belief_tracker_fn,
                tracker_fn_kwargs={
                    "num_agents": N_AGENTS,
                    "step_limit": ENV_STEP_LIMIT,
                    "discount": DISCOUNT,
                },
                renderer_fn=None,
                **vars(args),
            )
        )

    if args.get_num_exps:
        print(f"Number of experiments = {len(exp_params_list)}")
        return

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_str = "" if args.run_exp_id is None else f"_exp{args.run_exp_id}"
    exp_name = f"klr{exp_str}_{seed_str}"

    exp_args = vars(args)
    exp_args["env_id"] = ENV_ID
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

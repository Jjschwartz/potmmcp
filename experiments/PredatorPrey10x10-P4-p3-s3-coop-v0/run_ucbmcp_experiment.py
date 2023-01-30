"""Run BAPOSGMCP experiment in PP env with teams of SP agents.

Experiments
- IPOMCP (UCB + Meta-Policy)
  - Best meta-policy [Truncated]
  - Best meta-policy [UnTruncated]
- IPOMCP Random (UCB + Random) [Untruncated]

"""
import copy
from pprint import pprint

import baposgmcp.baselines as baseline_lib
import baposgmcp.run as run_lib

from run_experiment import (
    ENV_ID,
    NUM_SIMS,
    DISCOUNT,
    N_AGENTS,
    ENV_STEP_LIMIT,
    BAPOSGMCP_AGENT_ID,
    BAPOSGMCP_PUCT_KWARGS,
    UCB_C,
    POLICY_PRIOR_MAP,
    BEST_META_PI_MAP,
    BEST_META_PI_NAME
)

# DEBUG - Delete/comment this
# NUM_SIMS = [2, 5]


def get_ucb_mcps():  # noqa
    """
    - UCB MCP (UCB + Meta-Policy)
      - Best meta-policy [Truncated]
      - Best meta-policy [UnTruncated]
    - UCB MCP Random (UCB + Random) [Untruncated]
    """
    ucb_kwargs = copy.deepcopy(BAPOSGMCP_PUCT_KWARGS)
    ucb_kwargs["c"] = UCB_C
    ucb_kwargs["action_selection"] = "ucb"

    meta_variable_params = {"num_sims": NUM_SIMS, "truncated": [True]}
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
    # = 5 * 2 + 5
    # = 15
    assert len(ucb_params) == len(NUM_SIMS) * 2
    return ucb_params


def main(args):  # noqa
    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    # - IPOMCP (UCB + Meta-Policy)
    #   - Best meta-policy [Truncated]
    #   - Best meta-policy [UnTruncated]
    # - IPOMCP Random (UCB + Random) [Untruncated]
    policy_params = get_ucb_mcps()

    exp_params_list = []
    for pi_state in POLICY_PRIOR_MAP:
        assert len(pi_state) == N_AGENTS
        other_params = []
        for i, pi_id in enumerate(pi_state):
            if i == BAPOSGMCP_AGENT_ID:
                continue
            params_i = run_lib.load_posggym_agent_params([pi_id])
            other_params.append(params_i)

        pi_state_exp_params_list = run_lib.get_pairwise_exp_params(
            ENV_ID,
            [policy_params, *other_params],
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
        )
        exp_params_list.extend(pi_state_exp_params_list)

    if args.get_num_exps:
        print(f"Number of experiments = {len(exp_params_list)}")
        return

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_str = "" if args.run_exp_id is None else f"_exp{args.run_exp_id}"
    exp_name = f"baposgmcp{exp_str}_{seed_str}"

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

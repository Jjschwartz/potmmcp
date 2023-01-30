"""Run BAPOSGMCP experiment in PursuitEvasion env with KLR policies.

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
    BAPOSGMCP_PUCT_KWARGS,
    UCB_C,
    POLICY_PRIOR_MAP,
    BEST_META_PI_MAP,
    BEST_META_PI_NAME
)

# DEBUG - Delete/comment this
# NUM_SIMS = [2, 5]


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

        # - UCB MCP (UCB + Meta-Policy)
        #   - Best meta-policy [Truncated]
        # - UCB MCP Random (UCB + Random) [Untruncated]
        policy_params = get_ucb_mcps(i, j)

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

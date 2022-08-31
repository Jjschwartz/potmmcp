from pprint import pprint
from typing import Optional, List
from itertools import combinations_with_replacement

import posggym

from baposgmcp import pbt
import baposgmcp.run as run_lib
import baposgmcp.rllib as ba_rllib


def _get_klr_meta_policy_dict(policy_dir: str,
                              include_policy_ids: Optional[List[str]]):
    igraph = ba_rllib.import_igraph(policy_dir, True)
    meta_policy_dict = {}
    for policy_id in igraph.policies[igraph.SYMMETRIC_ID]:
        if (
            include_policy_ids is not None
            and policy_id not in include_policy_ids
        ):
            continue
        _, k = pbt.parse_klr_policy_id(policy_id)
        meta_policy_id = pbt.get_klr_policy_id(None, k+1, True)
        assert meta_policy_id in igraph.policies[igraph.SYMMETRIC_ID]
        meta_policy_dict[(-1, policy_id)] = {meta_policy_id: 1.0}
    return meta_policy_dict


def _get_baposgmcp_params(args):
    env = posggym.make(args.env_name)
    assert env.n_agents == 2
    baposgmcp_agent_id = 0

    episode_step_limit = env.spec.max_episode_steps

    baposgmcp_params = []
    for policy_dir in args.baposgmcp_policy_dirs:
        meta_policy_dict = _get_klr_meta_policy_dict(
            policy_dir, args.baposgmcp_policy_ids
        )

        baposgmcp_params.extend(run_lib.load_baposgmcp_params(
            args.env_name,
            baposgmcp_agent_id,
            args.gamma,
            args.num_sims,
            baposgmcp_kwargs={
                "c_init": 1.25,
                "c_base": 20000.0,
                "truncated": True,
                "extra_particles_prop": 1.0 / 16,
                "step_limit": episode_step_limit,
                "epsilon": 0.01,
            },
            other_policy_dir=policy_dir,
            other_policy_ids=args.baposgmcp_policy_ids,
            # uniform policy prior
            other_policy_dist=None,
            meta_policy_dir=policy_dir,
            meta_policy_dict=meta_policy_dict
        ))
    return baposgmcp_params


def main(args):   # noqa
    ba_rllib.register_posggym_env(args.env_name)
    env = posggym.make(args.env_name)
    assert env.n_agents == 2

    print("\n== Running Experiments ==")
    pprint(vars(args))

    print("== Creating Experiments ==")
    if args.other_policy_dirs is None:
        args.other_policy_dirs = args.baposgmcp_policy_dirs

    if args.other_policy_ids is None:
        args.other_policy_ids = args.baposgmcp_policy_ids

    other_policy_params = run_lib.load_all_agent_policy_params(
        args.env_name,
        args.other_policy_dirs,
        args.gamma,
        args.other_policy_ids,
        policy_load_kwargs=None
    )

    other_policy_params = list(
        combinations_with_replacement(other_policy_params, env.n_agents-1)
    )

    baposgmcp_params = _get_baposgmcp_params(args)

    exp_params_list = run_lib.get_baposgmcp_exp_params(
        baposgmcp_params=baposgmcp_params,
        other_policy_params=other_policy_params,
        **vars(args)
    )

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_name = f"baposgmcp_experiment_{seed_str}"

    print(f"== Running {len(exp_params_list)} Experiments ==")
    print(f"== Using {args.n_procs} CPUs ==")
    run_lib.run_experiments(
        exp_name,
        exp_params_list=exp_params_list[:2],
        exp_log_level=args.log_level,
        n_procs=args.n_procs,
        exp_args=vars(args),
        root_save_dir=args.root_save_dir
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = run_lib.get_baposgmcp_exp_parser()
    main(parser.parse_args())

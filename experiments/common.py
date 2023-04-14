"""Common classes and functions for the experiments.

Includes functionallity for running experiments with:
- POTMMCP (PUCB + Meta-Policy)
  - Greedy [Truncated]
  - Softmax [Truncated]
  - Uniform [Truncated]
- POTMMCP (PUCB + Random) [Untruncated]
- POTMMCP (PUCB + Fixed policies) [Truncated]
- IPOMCP (UCB + Meta-Policy)
  - Best meta-policy [Truncated]
- IPOMCP Random (UCB + Random) [Untruncated]
- Meta-Policy
  - Greedy
  - Softmax
  - Uniform

"""
import math
from dataclasses import dataclass, field
from itertools import product
from pprint import pprint
from typing import Any, Dict, List, Optional, Set, Tuple, Union, no_type_check

import pandas as pd

import potmmcp.baselines as baseline_lib
import potmmcp.plot as plot_utils
import potmmcp.policy as P
import potmmcp.run as run_lib
from potmmcp import meta_policy


@dataclass
class EnvExperimentParams:
    """Parameters for experiments."""

    env_id: str
    n_agents: int
    num_actions: int
    env_step_limit: int
    symmetric_env: bool

    # agent_id -> policy ids
    policy_ids: Dict[int, List[str]]
    policy_prior_map: Dict[P.PolicyState, float]
    pairwise_returns: Dict[P.PolicyState, Dict[P.PolicyID, float]]

    # For running many pi experiments
    many_pi_policy_ids: Optional[Dict[int, List[str]]] = None
    many_pi_policy_prior_map: Optional[Dict[P.PolicyState, float]] = None
    many_pi_pairwise_returns: Optional[
        Dict[P.PolicyState, Dict[P.PolicyID, float]]
    ] = None

    planning_agent_id: int = 0
    discount: float = 0.99
    epsilon: float = 0.01
    pucb_c: float = 1.25
    ucb_c: float = math.sqrt(2)  # as per OG paper/standard parameter
    dirichlet_alpha_denom: float = 10
    root_exploration_fraction: Union[float, List[float]] = 0.5
    reinvigorator = None  # Use default rejection sampler
    known_bounds = None
    extra_particles_prop: float = 1.0 / 16

    best_meta_pi_name: str = "softmax"
    softmax_temperatur: float = 0.25

    search_time_limits: List[float] = field(default_factory=lambda: [0.1, 1, 5, 10, 20])

    include_meta_baseline: bool = True
    include_ipomcp_baseline: bool = True

    def set_test_run(self):
        self.search_time_limits = [0.01, 0.05]

    def get_potmmcp_kwargs(self) -> Dict[str, Any]:
        if isinstance(self.root_exploration_fraction, list):
            # will be added as variable later, in get_potmmcp_params fn
            root_exploration_fraction = self.root_exploration_fraction[0]
        else:
            root_exploration_fraction = self.root_exploration_fraction

        return {
            "discount": self.discount,
            # use search_time_limit instead, added later
            "num_sims": None,
            "c": self.pucb_c,
            "truncated": True,
            "action_selection": "pucb",
            "dirichlet_alpha": self.num_actions / self.dirichlet_alpha_denom,
            "root_exploration_fraction": root_exploration_fraction,
            "reinvigorator": self.reinvigorator,
            "known_bounds": self.known_bounds,
            "extra_particles_prop": self.extra_particles_prop,
            "step_limit": self.env_step_limit,
            "epsilon": self.epsilon,
        }

    def get_potmmcp_random_kwargs(self) -> Dict[str, Any]:
        kwargs = self.get_potmmcp_kwargs()
        kwargs["truncated"] = False
        return kwargs

    def get_ipomcppf_meta_kwargs(self) -> Dict[str, Any]:
        kwargs = self.get_potmmcp_kwargs()
        kwargs["c"] = self.ucb_c
        kwargs["action_selection"] = "ucb"
        return kwargs

    def get_ipomcppf_random_kwargs(self) -> Dict[str, Any]:
        kwargs = self.get_ipomcppf_meta_kwargs()
        kwargs["truncated"] = False
        return kwargs

    def get_meta_policy_maps(
        self, best_only: bool = False, many_pi: bool = False
    ) -> Dict[str, Dict[P.PolicyState, P.PolicyDist]]:
        if many_pi:
            assert self.many_pi_pairwise_returns is not None
            pairwise_returns = self.many_pi_pairwise_returns
        else:
            pairwise_returns = self.pairwise_returns

        meta_policy_maps = {
            "greedy": meta_policy.get_greedy_policy_dict(pairwise_returns),
            "uniform": meta_policy.get_uniform_policy_dict(pairwise_returns),
            "softmax": meta_policy.get_softmax_policy_dict(
                pairwise_returns, self.softmax_temperatur
            ),
        }
        if best_only:
            return {self.best_meta_pi_name: meta_policy_maps[self.best_meta_pi_name]}
        return meta_policy_maps

    def get_agent_id_suffix(self) -> str:
        if self.symmetric_env:
            return ""
        return f"_i{self.planning_agent_id}"

    def get_other_agent_id_suffix(self, agent_id: int) -> str:
        if self.symmetric_env:
            return ""
        return f"_i{agent_id}"

    def get_meta_baseline_params(
        self, best_only: bool = False, many_pi: bool = False
    ) -> List[run_lib.PolicyParams]:
        """Get Meta-Policy experiment policy params.

        - Meta-Policy
          - Greedy
          - Softmax
          - Uniform

        """
        meta_policy_maps = self.get_meta_policy_maps(best_only)

        agent_id_suffix = self.get_agent_id_suffix()

        if many_pi:
            assert self.many_pi_policy_prior_map is not None
            policy_prior_map = self.many_pi_policy_prior_map
        else:
            policy_prior_map = self.policy_prior_map

        baseline_params = []
        for name, meta_policy_map in meta_policy_maps.items():
            # Meta Baseline Policy
            policy_id = f"metabaseline_{name}{agent_id_suffix}"
            policy_params = run_lib.PolicyParams(
                id=policy_id,
                entry_point=baseline_lib.MetaBaselinePolicy.posggym_agents_entry_point,
                kwargs={
                    "policy_id": policy_id,
                    "policy_prior_map": policy_prior_map,
                    "meta_policy_dict": meta_policy_map,
                },
            )
            baseline_params.append(policy_params)

        # Num exps:
        # = |Meta|
        assert len(baseline_params) == len(meta_policy_maps)
        return baseline_params

    def get_potmmcp_params(
        self,
        best_only: bool = False,
        include_random: bool = True,
        include_fixed: bool = True,
        many_pi: bool = False,
    ) -> List[run_lib.PolicyParams]:
        """Get POTMMCP experiment policy params.

        - POTMMCP (PUCB + Meta-Policy)
          - Greedy [Truncated]
          - Softmax [Truncated]
          - Uniform [Truncated]
        - POTMMCP (PUCB + Random) [Untruncated]
        - POTMMCP (PUCB + Fixed policies) [Truncated]

        Will also get params for different root_exploration_params if they are given
        as a list.

        """
        agent_id_suffix = self.get_agent_id_suffix()

        variable_params: Dict[str, List[Any]] = {
            "search_time_limit": self.search_time_limits,
            "truncated": [True],
        }

        if isinstance(self.root_exploration_fraction, list):
            variable_params[
                "root_exploration_fraction"
            ] = self.root_exploration_fraction

        if many_pi:
            assert self.many_pi_policy_prior_map is not None
            policy_prior_map = self.many_pi_policy_prior_map
        else:
            policy_prior_map = self.policy_prior_map

        meta_policy_maps = self.get_meta_policy_maps(best_only, many_pi)

        potmmcp_params = []
        for name, meta_policy_map in meta_policy_maps.items():
            potmmcp_params.extend(
                run_lib.load_potmmcp_params(
                    variable_params=variable_params,
                    potmmcp_kwargs=self.get_potmmcp_kwargs(),
                    policy_prior_map=policy_prior_map,
                    meta_policy_dict=meta_policy_map,
                    base_policy_id=f"potmmcp_meta{name}{agent_id_suffix}",
                )
            )
        if isinstance(self.root_exploration_fraction, list):
            # |TIMES| * |META| * |lambda|
            expected_num_params = (
                len(self.search_time_limits)
                * len(meta_policy_maps)
                * len(self.root_exploration_fraction)
            )
        else:
            # |TIMES| * |META|
            expected_num_params = len(self.search_time_limits) * len(meta_policy_maps)

        if include_random:
            variable_params["truncated"] = [False]
            if isinstance(self.root_exploration_fraction, list):
                # this has no effect since prior is uniform, so would be adding uniform
                # to uniform
                variable_params["root_exploration_fraction"] = [
                    self.root_exploration_fraction[0]
                ]

            potmmcp_params.extend(
                baseline_lib.load_random_potmmcp_params(
                    variable_params=variable_params,
                    potmmcp_kwargs=self.get_potmmcp_random_kwargs(),
                    policy_prior_map=policy_prior_map,
                    base_policy_id=f"potmmcp-random{agent_id_suffix}",
                )
            )
            # |TIMES|
            expected_num_params += len(self.search_time_limits)

        if include_fixed:
            variable_params["truncated"] = [True]
            if isinstance(self.root_exploration_fraction, list):
                variable_params[
                    "root_exploration_fraction"
                ] = self.root_exploration_fraction

            fixed_policy_ids = self.policy_ids[self.planning_agent_id]
            potmmcp_params.extend(
                baseline_lib.load_fixed_pi_potmmcp_params(
                    variable_params=variable_params,
                    fixed_policy_ids=fixed_policy_ids,
                    potmmcp_kwargs=self.get_potmmcp_kwargs(),
                    policy_prior_map=policy_prior_map,
                    base_policy_id=f"potmmcp-fixed{agent_id_suffix}",
                )
            )
            if isinstance(self.root_exploration_fraction, list):
                # |TIMES| * |PI| * |lambda|
                expected_num_params += (
                    len(self.search_time_limits)
                    * len(fixed_policy_ids)
                    * len(self.root_exploration_fraction)
                )
            else:
                # |TIMES| * |PI|
                expected_num_params += len(self.search_time_limits) * len(
                    fixed_policy_ids
                )

        # Num exps:
        assert len(potmmcp_params) == expected_num_params
        return potmmcp_params

    def get_ipomcppf_params(self) -> List[run_lib.PolicyParams]:
        """Get I-POMCP-PF experiment policy params.

        - I-POMCP-PF + Best Meta (UCB + Meta-Policy) [Truncated]
        - I-POMCP-PF + Random (UCB + Random) [Untruncated]

        """
        agent_id_suffix = self.get_agent_id_suffix()

        meta_policy_maps = self.get_meta_policy_maps(True)

        ipomcppf_params = run_lib.load_potmmcp_params(
            variable_params={
                "search_time_limit": self.search_time_limits,
                "truncated": [True],
            },
            potmmcp_kwargs=self.get_ipomcppf_meta_kwargs(),
            policy_prior_map=self.policy_prior_map,
            meta_policy_dict=meta_policy_maps[self.best_meta_pi_name],
            base_policy_id=f"ucbmcp_meta{self.best_meta_pi_name}{agent_id_suffix}",
        )

        ipomcppf_params.extend(
            baseline_lib.load_random_potmmcp_params(
                variable_params={
                    "search_time_limit": self.search_time_limits,
                    "truncated": [False],
                },
                potmmcp_kwargs=self.get_ipomcppf_random_kwargs(),
                policy_prior_map=self.policy_prior_map,
                base_policy_id=f"ucbmcp-random{agent_id_suffix}",
            )
        )

        # NUM Exps:
        # = |SEARCH_TIME_LIMITS| + |SEARCH_TIME_LIMITS|
        # = 5 + 5
        # = 10
        assert len(ipomcppf_params) == len(self.search_time_limits) * 2
        return ipomcppf_params

    def get_other_agent_policy_params(
        self, many_pi: bool = False
    ) -> List[List[List[run_lib.PolicyParams]]]:
        """Get other agent experiment policy params.

        Each entry is a joint policy for the other agents (i.e. one policy per agent).
        The entries are in order of the policy ID of the other agent, with the joint
        policy entry for the planning agent missing.

        """
        if many_pi:
            assert self.many_pi_policy_prior_map is not None
            policy_prior_map = self.many_pi_policy_prior_map
        else:
            policy_prior_map = self.policy_prior_map

        other_params = []
        for pi_state in policy_prior_map:
            assert len(pi_state) == self.n_agents
            joint_pi = []
            for i, pi_id in enumerate(pi_state):
                if i == self.planning_agent_id:
                    continue
                joint_pi.append(run_lib.load_posggym_agent_params([pi_id]))
            other_params.append(joint_pi)
        return other_params

    def get_other_agent_mixed_policy_params(
        self, many_pi: bool
    ) -> List[List[run_lib.PolicyParams]]:
        """Get other agent experiment mixed policy params.

        The mixed policy is a single policy which selects a new policy to use each
        episode based on a distribution.

        Uses a uniform distribution over the available policies for each agent.

        Outputs a joint policy for the other agents (i.e. one mixed policy per agent).
        The entries are in order of the policy ID of the other agent, with the joint
        policy entry for the planning agent missing.

        """
        if many_pi:
            assert self.many_pi_policy_prior_map is not None
            policy_prior_map = self.many_pi_policy_prior_map
        else:
            policy_prior_map = self.policy_prior_map

        policies: Dict[int, Set[str]] = {
            i: set() for i in range(self.n_agents) if i != self.planning_agent_id
        }
        for pi_state in policy_prior_map:
            assert len(pi_state) == self.n_agents
            for i, pi_id in enumerate(pi_state):
                if i == self.planning_agent_id:
                    continue
                policies[i].add(pi_id)

        policy_dist = {}
        for i, pi_ids in policies.items():
            prob = 1.0 / len(pi_ids)
            policy_dist[i] = {pi_id: prob for pi_id in pi_ids}

        joint_pi = []
        for i, pi_dist in policy_dist.items():
            if i == self.planning_agent_id:
                continue
            agent_id_suffix = self.get_other_agent_id_suffix(i)
            policy_id = f"mixed{agent_id_suffix}"
            policy_params = run_lib.PolicyParams(
                id=policy_id,
                entry_point=baseline_lib.MixedPolicy.posggym_agents_entry_point,
                kwargs={
                    "policy_id": policy_id,
                    "policy_dist": pi_dist,
                },
            )
            joint_pi.append([policy_params])
        return joint_pi

    def get_experiments(
        self,
        init_seed: int,
        num_seeds: int,
        num_episodes: int,
        time_limit: Optional[int] = None,
        record_env: bool = False,
    ) -> List[run_lib.ExpParams]:
        """Get list of params for each individual experiment run."""
        planning_params = self.get_potmmcp_params(
            best_only=False, include_random=True, include_fixed=True, many_pi=False
        )
        if self.include_ipomcp_baseline:
            planning_params.extend(self.get_ipomcppf_params())
        if self.include_meta_baseline:
            planning_params.extend(
                self.get_meta_baseline_params(best_only=False, many_pi=False)
            )

        other_params = self.get_other_agent_policy_params(many_pi=False)
        print(f"Number of planning agent params = {len(planning_params)}.")
        print(f"Number of other agent params = {len(other_params)}.")

        exp_params_list: List[run_lib.ExpParams] = []
        for joint_policy in other_params:
            joint_policy.insert(self.planning_agent_id, planning_params)
            exp_params = run_lib.get_pairwise_exp_params(
                env_id=self.env_id,
                policy_params=joint_policy,
                init_seed=init_seed,
                num_seeds=num_seeds,
                num_episodes=num_episodes,
                discount=self.discount,
                time_limit=time_limit,
                exp_id_init=len(exp_params_list),
                tracker_fn=run_lib.belief_tracker_fn,
                tracker_fn_kwargs={
                    "num_agents": self.n_agents,
                    "step_limit": self.env_step_limit,
                    "discount": self.discount,
                },
                renderer_fn=None,
                record_env=record_env,
            )
            exp_params_list.extend(exp_params)

        return exp_params_list

    def get_lambda_experiments(
        self,
        init_seed: int,
        num_seeds: int,
        num_episodes: int,
        time_limit: Optional[int] = None,
        record_env: bool = False,
    ) -> List[run_lib.ExpParams]:
        """Get list of params for each individual experiment run."""
        self.root_exploration_fraction = [0.0, 1 / 4, 1 / 3, 1 / 2]
        self.include_ipomcp_baseline = False
        self.include_meta_baseline = False

        if self.search_time_limits == [0.1, 1, 5, 10, 20]:
            # only change if using default
            self.search_time_limits = [0.1, 1, 5, 10]

        planning_params = self.get_potmmcp_params(
            best_only=True, include_fixed=False, include_random=False, many_pi=False
        )

        other_params = self.get_other_agent_policy_params()
        print(f"Number of planning agent params = {len(planning_params)}.")
        print(f"Number of other agent params = {len(other_params)}.")

        exp_params_list: List[run_lib.ExpParams] = []
        for joint_policy in other_params:
            joint_policy.insert(self.planning_agent_id, planning_params)
            exp_params = run_lib.get_pairwise_exp_params(
                env_id=self.env_id,
                policy_params=joint_policy,
                init_seed=init_seed,
                num_seeds=num_seeds,
                num_episodes=num_episodes,
                discount=self.discount,
                time_limit=time_limit,
                exp_id_init=len(exp_params_list),
                tracker_fn=run_lib.belief_tracker_fn,
                tracker_fn_kwargs={
                    "num_agents": self.n_agents,
                    "step_limit": self.env_step_limit,
                    "discount": self.discount,
                },
                renderer_fn=None,
                record_env=record_env,
            )
            exp_params_list.extend(exp_params)

        return exp_params_list

    def get_many_pi_experiments(
        self,
        init_seed: int,
        num_seeds: int,
        num_episodes: int,
        time_limit: Optional[int] = None,
        record_env: bool = False,
    ) -> List[run_lib.ExpParams]:
        """Get list of params for each individual experiment run."""
        self.include_ipomcp_baseline = False
        self.include_meta_baseline = True

        if self.search_time_limits == [0.1, 1, 5, 10, 20]:
            # only change if using default
            self.search_time_limits = [0.1, 1, 5, 10]

        print("Many Pi policy params")
        print("\nmany_pi_policy_ids")
        pprint(self.many_pi_policy_ids)

        print("\nmany_pi_policy_prior_map")
        pprint(self.many_pi_policy_prior_map)

        print("\nmany_pi_policy_pairwise_returns")
        pprint(self.many_pi_pairwise_returns)

        planning_params = self.get_potmmcp_params(
            best_only=True, include_fixed=False, include_random=False, many_pi=True
        )
        if self.include_meta_baseline:
            planning_params.extend(
                self.get_meta_baseline_params(best_only=True, many_pi=True)
            )

        other_params = self.get_other_agent_mixed_policy_params(many_pi=True)
        print(f"\nNumber of planning agent params = {len(planning_params)}.")
        print(f"Number of other agent params = {len(other_params)}.")

        exp_params_list: List[run_lib.ExpParams] = []
        joint_policy = other_params

        joint_policy.insert(self.planning_agent_id, planning_params)
        exp_params = run_lib.get_pairwise_exp_params(
            env_id=self.env_id,
            policy_params=joint_policy,
            init_seed=init_seed,
            num_seeds=num_seeds,
            num_episodes=num_episodes,
            discount=self.discount,
            time_limit=time_limit,
            exp_id_init=len(exp_params_list),
            tracker_fn=run_lib.belief_tracker_fn,
            tracker_fn_kwargs={
                "num_agents": self.n_agents,
                "step_limit": self.env_step_limit,
                "discount": self.discount,
            },
            renderer_fn=None,
            record_env=record_env,
        )
        exp_params_list.extend(exp_params)

        return exp_params_list

    def get_other_policy_ids(
        self, remove_env_id: bool = False, many_pi: bool = False
    ) -> List[str]:
        if many_pi:
            assert self.many_pi_policy_ids is not None
            policy_ids_map = self.many_pi_policy_ids
        else:
            policy_ids_map = self.policy_ids

        if self.symmetric_env:
            policy_ids = policy_ids_map[self.planning_agent_id]
        else:
            assert self.n_agents == 2
            policy_ids = policy_ids_map[(self.planning_agent_id + 1) % 2]

        if not remove_env_id:
            return policy_ids
        return [pi_id.split("/")[1] for pi_id in policy_ids]

    def get_planning_policy_ids(
        self, remove_env_id: bool = False, many_pi: bool = False
    ) -> List[str]:
        if many_pi:
            assert self.many_pi_policy_ids is not None
            policy_ids_map = self.many_pi_policy_ids
        else:
            policy_ids_map = self.policy_ids

        policy_ids = policy_ids_map[self.planning_agent_id]
        if not remove_env_id:
            return policy_ids
        return [pi_id.split("/")[1] for pi_id in policy_ids]

    def get_all_policy_ids(
        self, remove_env_id: bool = False, many_pi: bool = False
    ) -> List[str]:
        if many_pi:
            assert self.many_pi_policy_ids is not None
            policy_ids_map = self.many_pi_policy_ids
        else:
            policy_ids_map = self.policy_ids

        policy_ids_set: Set[str] = set()
        for pi_ids in policy_ids_map.values():
            policy_ids_set.update(pi_ids)

        policy_ids = list(policy_ids_set)
        policy_ids.sort()

        if not remove_env_id:
            return policy_ids
        return [pi_id.split("/")[1] for pi_id in policy_ids]

    def get_policy_prior_map(
        self, remove_env_id: bool = False, many_pi: bool = False
    ) -> Dict[P.PolicyState, float]:
        if many_pi:
            assert self.many_pi_policy_prior_map is not None
            policy_prior_map = self.many_pi_policy_prior_map
        else:
            policy_prior_map = self.policy_prior_map

        if not remove_env_id:
            return policy_prior_map

        prior_map = {}
        for pi_state, prob in policy_prior_map.items():
            new_pi_state = []
            for pi_id in pi_state:
                if pi_id == "-1":
                    new_pi_state.append(pi_id)
                else:
                    new_pi_state.append(pi_id.split("/")[1])
            prior_map[tuple(new_pi_state)] = prob
        return prior_map

    def get_other_joint_policies(
        self, remove_env_id: bool = False, many_pi: bool = False
    ) -> List[P.PolicyState]:
        prior_map = self.get_policy_prior_map(remove_env_id, many_pi)

        joint_pis = []
        for pi_state in prior_map:
            other_pi_state = []
            for pi_id in pi_state:
                if pi_id != "-1":
                    other_pi_state.append(pi_id)
            joint_pis.append(tuple(other_pi_state))
        return joint_pis


def run_env_experiments(env_exp_params: EnvExperimentParams):
    parser = run_lib.get_exp_parser()
    parser.add_argument(
        "--get_num_exps",
        action="store_true",
        help="Compute and display number of experiments without running them.",
    )
    parser.add_argument(
        "--run_lambda_experiment",
        action="store_true",
        help="Run lambda experiment, instead of main experiment.",
    )
    parser.add_argument(
        "--run_many_pi_experiment",
        action="store_true",
        help="Run Many pi experiment, instead of main/lambda experiment.",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Do a test run with reduced # and time for exps.",
    )
    parser.add_argument(
        "--enumerate_exps",
        action="store_true",
        help=(
            "Enumerate the experiments without running them. Useful for finding ID of "
            "specific experiment, e.g. for recording video."
        ),
    )
    args = parser.parse_args()

    print("\n== Running Experiments ==")
    pprint(vars(args))

    if args.test_run:
        env_exp_params.set_test_run()

    print("== Creating Experiments ==")
    if args.run_lambda_experiment:
        exp_params_list = env_exp_params.get_lambda_experiments(
            args.init_seed,
            args.num_seeds,
            args.num_episodes,
            args.time_limit,
            args.record_env,
        )
    elif args.run_many_pi_experiment:
        exp_params_list = env_exp_params.get_many_pi_experiments(
            args.init_seed,
            args.num_seeds,
            args.num_episodes,
            args.time_limit,
            args.record_env,
        )
    else:
        exp_params_list = env_exp_params.get_experiments(
            args.init_seed,
            args.num_seeds,
            args.num_episodes,
            args.time_limit,
            args.record_env,
        )

    if args.enumerate_exps:
        for p in exp_params_list:
            print(p)
            input("Press enter for next experiment")
        return

    if args.get_num_exps:
        print(f"Number of experiments = {len(exp_params_list)}")
        return

    seed_str = f"initseed{args.init_seed}_numseeds{args.num_seeds}"
    exp_str = "" if args.run_exp_id is None else f"_exp{args.run_exp_id}"
    exp_name = f"potmmcp{exp_str}_{seed_str}"

    exp_args = vars(args)
    exp_args["env_id"] = env_exp_params.env_id
    exp_args["discount"] = env_exp_params.discount
    exp_args["potmmcp_kwargs"] = env_exp_params.get_potmmcp_kwargs()

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


@no_type_check
def get_policy_set_values(
    policy_results_file: str,
    env_id: str,
    planning_agent_id: int,
    env_symmetric: bool,
    excluded_policy_prefixes: List[str],
    excluded_other_policy_prefixes: List[str],
) -> Tuple[
    Dict[int, List[str]],
    Dict[P.PolicyState, float],
    Dict[P.PolicyState, Dict[P.PolicyID, float]],
]:
    """Get policy set values for an experiments.

    Returns
    -------
    policy_ids
    policy_prior_map
    pairwise_returns

    """
    df = plot_utils.import_results(policy_results_file)

    if env_symmetric:
        # for symmetric env we need to make sure there is a row for planning agent for
        # all (policy_id, co_team_id)
        # in asymmetric env, just drop rows for non-planning agent
        next_exp_id = df["exp_id"].max() + 1
        new_rows = []
        for pi_id, co_team_id in product(
            df["policy_id"].unique(), df["co_team_id"].unique()
        ):
            pair_df = df[(df["policy_id"] == pi_id) & (df["co_team_id"] == co_team_id)]
            if len(pair_df) == 0:
                print(f"missing entry for ({pi_id}, {co_team_id}")
                continue
            elif planning_agent_id in pair_df["agent_id"].unique():
                # already in df with correct agent id
                continue
            # take first entry (there should only be one)
            pair_row = df.loc[
                (df["policy_id"] == pi_id) & (df["co_team_id"] == co_team_id)
            ].copy()
            pair_row["agent_id"] = planning_agent_id
            pair_row["exp_id"] = next_exp_id
            next_exp_id += 1
            new_rows.append(pair_row)

        if len(new_rows):
            print(f"\nAdding {len(new_rows)} rows for agent_id={planning_agent_id}")
            new_pairs_df = pd.concat(new_rows, axis="rows").reset_index(drop=True)
            df = pd.concat([df, new_pairs_df], ignore_index=True)

    # Drop rows for non-planning agent
    df = df[df["agent_id"] == planning_agent_id]

    # drop policies we don't want
    for pi_id_prefix in excluded_policy_prefixes:
        df = df[~df["policy_id"].str.startswith(pi_id_prefix)]
        for column in df.columns:
            if column.startswith("coplayer_policy"):
                df = df[~df[column].str.startswith(pi_id_prefix)]

    # convert into empiriacal pairwise returns map
    col_ids = df["co_team_id"].unique().tolist()
    col_ids.sort()
    row_ids = df["policy_id"].unique().tolist()
    row_ids.sort()

    pairwise_returns: Dict[P.PolicyState, Dict[P.PolicyID, float]] = {}
    for co_team_id in col_ids:
        team_pis = list(co_team_id)
        if any(
            any(pi_id.startswith(prefix) for prefix in excluded_other_policy_prefixes)
            for pi_id in team_pis
        ):
            continue

        team_pis = [f"{env_id}/{co_pi_id}" for co_pi_id in co_team_id]
        team_pis.insert(planning_agent_id, "-1")
        pi_state = tuple(team_pis)

        pairwise_returns[pi_state] = {}
        for pi_id in row_ids:
            mean_return = df[
                (df["policy_id"] == pi_id) & (df["co_team_id"] == co_team_id)
            ]["episode_return_mean"].values[0]

            pairwise_returns[pi_state][f"{env_id}/{pi_id}"] = mean_return

    policy_ids = {planning_agent_id: [f"{env_id}/{pi_id}" for pi_id in row_ids]}

    policy_prior_map = {
        pi_state: 1.0 / len(pairwise_returns) for pi_state in pairwise_returns
    }

    return policy_ids, policy_prior_map, pairwise_returns

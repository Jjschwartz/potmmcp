"""Run POTMMCP with the different meta-policies in Driving env with KLR policies."""
import os.path as osp

import posggym_agents
from common import EnvExperimentParams, get_policy_set_values, run_env_experiments


ENV_ID = "Driving14x14WideRoundAbout-n2-v0"
PLANNING_AGENT_ID = 0
POLICY_SEED = 0

POSGGYM_AGENTS_POLICY_RESULTS_FILE = osp.join(
    posggym_agents.config.BASE_DIR,
    "agents",
    "driving14x14wideroundabout_n2_v0",
    "results",
    "klrbr_results.csv",
)

(
    many_pi_policy_ids,
    many_pi_policy_prior_map,
    many_pi_pairwise_returns,
) = get_policy_set_values(
    POSGGYM_AGENTS_POLICY_RESULTS_FILE,
    ENV_ID,
    PLANNING_AGENT_ID,
    env_symmetric=True,
    excluded_policy_prefixes=["klrbr", "uniform_random"],
    excluded_other_policy_prefixes=["klr_k4_seed"],
)


DRIVING_EXP_PARAMS = EnvExperimentParams(
    env_id=ENV_ID,
    n_agents=2,
    num_actions=5,
    env_step_limit=50,
    symmetric_env=True,
    policy_ids={
        PLANNING_AGENT_ID: [
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0",
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0",
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0",
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0",
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0",
        ]
    },
    # NOTE There are 5 policies available (K=4) but we assume the other agent is
    # only using 4 (K=3), but we have all (K=4) policies available for the meta policy
    policy_prior_map={
        ("-1", f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0"): 1 / 4,
        ("-1", f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0"): 1 / 4,
        ("-1", f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0"): 1 / 4,
        ("-1", f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0"): 1 / 4,
    },
    pairwise_returns={
        ("-1", f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 1.91,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 2.21,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 2.01,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 1.74,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0": 1.70,
        },
        ("-1", f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 2.21,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 2.05,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 2.21,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 1.74,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0": 1.88,
        },
        ("-1", f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 1.97,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 2.19,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 2.12,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 2.18,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0": 1.76,
        },
        ("-1", f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}-v0": 1.73,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}-v0": 1.76,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}-v0": 2.18,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}-v0": 1.89,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}-v0": 2.21,
        },
    },
    planning_agent_id=PLANNING_AGENT_ID,
    many_pi_policy_ids=many_pi_policy_ids,
    many_pi_policy_prior_map=many_pi_policy_prior_map,
    many_pi_pairwise_returns=many_pi_pairwise_returns
)


if __name__ == "__main__":
    run_env_experiments(DRIVING_EXP_PARAMS)

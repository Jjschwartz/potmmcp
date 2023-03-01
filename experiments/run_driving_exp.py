"""Run POTMMCP with the different meta-policies in Driving env with KLR policies."""
from common import EnvExperimentParams, run_env_experiments


ENV_ID = "Driving14x14WideRoundAbout-n2-v0"
PLANNING_AGENT_ID = 0
POLICY_SEED = 0

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
)


if __name__ == "__main__":
    run_env_experiments(DRIVING_EXP_PARAMS)

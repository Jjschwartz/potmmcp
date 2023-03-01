"""Run POTMMCP experiment in PursuitEvasion env with KLR policies."""
from common import EnvExperimentParams, run_env_experiments


ENV_ID = "PursuitEvasion16x16-v0"
PLANNING_AGENT_ID = 0   # Evader
POLICY_SEED = 0

PE_EVADER_EXP_PARAMS = EnvExperimentParams(
    env_id=ENV_ID,
    n_agents=2,
    num_actions=4,
    env_step_limit=100,
    symmetric_env=False,
    policy_ids={
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
    },
    # NOTE There are 5 policies available (K=4) but we assume the other agent is
    # only using 4 (K=3), but we have all (K=4) policies available for the
    # meta policy
    policy_prior_map={
        ("-1", f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0"): 1 / 4,
        ("-1", f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0"): 1 / 4,
        ("-1", f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0"): 1 / 4,
        ("-1", f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0"): 1 / 4,
    },
    # Defined for policy states with (K=3) and meta-policy (K=4)
    pairwise_returns={
        ("-1", f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": 0.21,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": 0.97,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": -0.16,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": -0.27,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": 0.60,
        },
        ("-1", f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": -0.66,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": -0.25,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": 0.91,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": 0.02,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": -0.41,
        },
        ("-1", f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": -0.60,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": -0.74,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": 0.25,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": 0.61,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": -0.20,
        },
        ("-1", f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": 0.19,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": 0.59,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": -0.78,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": -0.45,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": 0.97,
        },
        ("-1", f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i1-v0"): {
            f"{ENV_ID}/klr_k0_seed{POLICY_SEED}_i0-v0": 0.17,
            f"{ENV_ID}/klr_k1_seed{POLICY_SEED}_i0-v0": 0.67,
            f"{ENV_ID}/klr_k2_seed{POLICY_SEED}_i0-v0": -0.36,
            f"{ENV_ID}/klr_k3_seed{POLICY_SEED}_i0-v0": -0.78,
            f"{ENV_ID}/klr_k4_seed{POLICY_SEED}_i0-v0": 0.68,
        },
    },
    planning_agent_id=PLANNING_AGENT_ID
)


if __name__ == "__main__":
    run_env_experiments(PE_EVADER_EXP_PARAMS)

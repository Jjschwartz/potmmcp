"""Run POTMMCP experiment in PP env with teams of SP agents."""
from common import EnvExperimentParams, run_env_experiments


ENV_ID = "PredatorPrey10x10-P2-p3-s2-coop-v0"
PLANNING_AGENT_ID = 0

PP2_EXP_PARAMS = EnvExperimentParams(
    env_id=ENV_ID,
    n_agents=2,
    num_actions=5,
    env_step_limit=50,
    symmetric_env=True,
    policy_ids={
        PLANNING_AGENT_ID: [
            f"{ENV_ID}/sp_seed0-v0",
            f"{ENV_ID}/sp_seed1-v0",
            f"{ENV_ID}/sp_seed2-v0",
            f"{ENV_ID}/sp_seed3-v0",
            f"{ENV_ID}/sp_seed4-v0",
        ]
    },
    policy_prior_map={
        ("-1", f"{ENV_ID}/sp_seed0-v0"): 0.2,
        ("-1", f"{ENV_ID}/sp_seed1-v0"): 0.2,
        ("-1", f"{ENV_ID}/sp_seed2-v0"): 0.2,
        ("-1", f"{ENV_ID}/sp_seed3-v0"): 0.2,
        ("-1", f"{ENV_ID}/sp_seed4-v0"): 0.2,
    },
    pairwise_returns={
        ("-1", f"{ENV_ID}/sp_seed0-v0"): {
            f"{ENV_ID}/sp_seed0-v0": 0.98,
            f"{ENV_ID}/sp_seed1-v0": 0.29,
            f"{ENV_ID}/sp_seed2-v0": 0.47,
            f"{ENV_ID}/sp_seed3-v0": 0.32,
            f"{ENV_ID}/sp_seed4-v0": 0.64,
        },
        ("-1", f"{ENV_ID}/sp_seed1-v0"): {
            f"{ENV_ID}/sp_seed0-v0": 0.29,
            f"{ENV_ID}/sp_seed1-v0": 0.98,
            f"{ENV_ID}/sp_seed2-v0": 0.31,
            f"{ENV_ID}/sp_seed3-v0": 0.30,
            f"{ENV_ID}/sp_seed4-v0": 0.56,
        },
        ("-1", f"{ENV_ID}/sp_seed2-v0"): {
            f"{ENV_ID}/sp_seed0-v0": 0.47,
            f"{ENV_ID}/sp_seed1-v0": 0.31,
            f"{ENV_ID}/sp_seed2-v0": 0.99,
            f"{ENV_ID}/sp_seed3-v0": 0.71,
            f"{ENV_ID}/sp_seed4-v0": 0.37,
        },
        ("-1", f"{ENV_ID}/sp_seed3-v0"): {
            f"{ENV_ID}/sp_seed0-v0": 0.32,
            f"{ENV_ID}/sp_seed1-v0": 0.30,
            f"{ENV_ID}/sp_seed2-v0": 0.71,
            f"{ENV_ID}/sp_seed3-v0": 0.99,
            f"{ENV_ID}/sp_seed4-v0": 0.38,
        },
        ("-1", f"{ENV_ID}/sp_seed4-v0"): {
            f"{ENV_ID}/sp_seed0-v0": 0.64,
            f"{ENV_ID}/sp_seed1-v0": 0.56,
            f"{ENV_ID}/sp_seed2-v0": 0.37,
            f"{ENV_ID}/sp_seed3-v0": 0.38,
            f"{ENV_ID}/sp_seed4-v0": 0.99,
        },
    },
    planning_agent_id=PLANNING_AGENT_ID,
)


if __name__ == "__main__":
    run_env_experiments(PP2_EXP_PARAMS)

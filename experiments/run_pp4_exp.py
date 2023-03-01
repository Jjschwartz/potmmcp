"""Run POTMMCP experiment in PP env with teams of SP agents."""
from common import EnvExperimentParams, run_env_experiments


ENV_ID = "PredatorPrey10x10-P4-p3-s3-coop-v0"
PLANNING_AGENT_ID = 0

joint_pis = [
    ("-1", f"{ENV_ID}/sp_seed0-v0", f"{ENV_ID}/sp_seed0-v0", f"{ENV_ID}/sp_seed0-v0"),
    ("-1", f"{ENV_ID}/sp_seed1-v0", f"{ENV_ID}/sp_seed1-v0", f"{ENV_ID}/sp_seed1-v0"),
    ("-1", f"{ENV_ID}/sp_seed2-v0", f"{ENV_ID}/sp_seed2-v0", f"{ENV_ID}/sp_seed2-v0"),
    ("-1", f"{ENV_ID}/sp_seed3-v0", f"{ENV_ID}/sp_seed3-v0", f"{ENV_ID}/sp_seed3-v0"),
    ("-1", f"{ENV_ID}/sp_seed4-v0", f"{ENV_ID}/sp_seed4-v0", f"{ENV_ID}/sp_seed4-v0"),
]

PP4_EXP_PARAMS = EnvExperimentParams(
    env_id=ENV_ID,
    n_agents=4,
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
    policy_prior_map={pi: 1.0 / len(joint_pis) for pi in joint_pis},
    pairwise_returns={
        joint_pis[0]: {
            f"{ENV_ID}/sp_seed0-v0": 0.82,
            f"{ENV_ID}/sp_seed1-v0": 0.57,
            f"{ENV_ID}/sp_seed2-v0": 0.54,
            f"{ENV_ID}/sp_seed3-v0": 0.55,
            f"{ENV_ID}/sp_seed4-v0": 0.61,
        },
        joint_pis[1]: {
            f"{ENV_ID}/sp_seed0-v0": 0.17,
            f"{ENV_ID}/sp_seed1-v0": 0.96,
            f"{ENV_ID}/sp_seed2-v0": 0.29,
            f"{ENV_ID}/sp_seed3-v0": 0.49,
            f"{ENV_ID}/sp_seed4-v0": 0.35,
        },
        joint_pis[2]: {
            f"{ENV_ID}/sp_seed0-v0": 0.62,
            f"{ENV_ID}/sp_seed1-v0": 0.63,
            f"{ENV_ID}/sp_seed2-v0": 0.95,
            f"{ENV_ID}/sp_seed3-v0": 0.62,
            f"{ENV_ID}/sp_seed4-v0": 0.77,
        },
        joint_pis[3]: {
            f"{ENV_ID}/sp_seed0-v0": 0.53,
            f"{ENV_ID}/sp_seed1-v0": 0.56,
            f"{ENV_ID}/sp_seed2-v0": 0.57,
            f"{ENV_ID}/sp_seed3-v0": 0.93,
            f"{ENV_ID}/sp_seed4-v0": 0.54,
        },
        joint_pis[4]: {
            f"{ENV_ID}/sp_seed0-v0": 0.46,
            f"{ENV_ID}/sp_seed1-v0": 0.59,
            f"{ENV_ID}/sp_seed2-v0": 0.64,
            f"{ENV_ID}/sp_seed3-v0": 0.60,
            f"{ENV_ID}/sp_seed4-v0": 0.94,
        },
    },
    planning_agent_id=PLANNING_AGENT_ID,
)


if __name__ == "__main__":
    run_env_experiments(PP4_EXP_PARAMS)

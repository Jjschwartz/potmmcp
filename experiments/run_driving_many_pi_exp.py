"""Run POTMMCP with the different meta-policies in Driving env with KLR policies."""
import os.path as osp

import posggym_agents
from common import EnvExperimentParams, run_env_experiments, get_policy_set_values

import potmmcp.plot as plot_utils


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
policy_df = plot_utils.import_results(POSGGYM_AGENTS_POLICY_RESULTS_FILE)

policy_ids, policy_prior_map, pairwise_returns = get_policy_set_values(
    POSGGYM_AGENTS_POLICY_RESULTS_FILE,
    ENV_ID,
    PLANNING_AGENT_ID,
    env_symmetric=True,
    excluded_policy_prefixes=["klrbr", "uniform_random"],
    excluded_other_policy_prefixes=["klr_k4_seed"]
)


DRIVING_MANY_EXP_PARAMS = EnvExperimentParams(
    env_id=ENV_ID,
    n_agents=2,
    num_actions=5,
    env_step_limit=50,
    symmetric_env=True,
    policy_ids=policy_ids,
    # NOTE There are 5 policies available (K=4) but we assume the other agent is
    # only using 4 (K=3), but we have all (K=4) policies available for the meta policy
    policy_prior_map=policy_prior_map,
    pairwise_returns=pairwise_returns,
    planning_agent_id=PLANNING_AGENT_ID,
)


if __name__ == "__main__":
    run_env_experiments(DRIVING_MANY_EXP_PARAMS)

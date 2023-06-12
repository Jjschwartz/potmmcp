import os
import os.path as osp
import pickle

import posggym.model as M
from gym import spaces
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from posggym_agents.agents.registration import PolicySpec
from posggym_agents.rllib import PPORllibPolicy, get_flatten_preprocessor


ENV_ID = "PredatorPrey10x10-P2-p3-s2-coop-v0"
BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_AGENT_DIR = osp.join(BASE_DIR, "agents")

# Map from id to policy spec for this env
POLICY_SPECS = {}


def load_rllib_policy_spec(id: str, policy_file: str) -> PolicySpec:
    """Load policy spec for from policy dir.

    'id' is the unique ID for the policy to be used in the global registry
    'policy_file' is the path to the agent .pkl file.

    """

    def _entry_point(model: M.POSGModel, agent_id: M.AgentID, policy_id: str, **kwargs):
        preprocessor = get_flatten_preprocessor(model.observation_spaces[agent_id])

        with open(osp.join(BASE_AGENT_DIR, policy_file), "rb") as f:
            data = pickle.load(f)

        ppo_policy = PPOTorchPolicy(
            spaces.flatten_space(model.observation_spaces[agent_id]),
            model.action_spaces[agent_id],
            data["config"],
        )
        ppo_policy.set_state(data["state"])

        return PPORllibPolicy(model, agent_id, policy_id, ppo_policy, preprocessor)

    return PolicySpec(id, _entry_point)


for policy_file in os.listdir(BASE_AGENT_DIR):
    # remove ".pkl"
    policy_id = policy_file.split(".")[0]
    # unique ID used in posggym-agents global registry
    id = f"{ENV_ID}/{policy_id}-v0"
    POLICY_SPECS[id] = load_rllib_policy_spec(id, policy_file)

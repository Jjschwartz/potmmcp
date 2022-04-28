import pathlib
import os.path as osp

import posggym
from posggym.wrappers import FlattenObservation
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv


EXP_BASE_DIR = osp.dirname(osp.abspath(__file__))
EXP_RESULTS_DIR = osp.join(EXP_BASE_DIR, "results")
EXP_SAVED_RESULTS_DIR = osp.join(EXP_BASE_DIR, "saved_results")
EXP_RL_POLICY_DIR = osp.join(EXP_BASE_DIR, "rl_policies")

pathlib.Path(EXP_RESULTS_DIR).mkdir(exist_ok=True)
pathlib.Path(EXP_RL_POLICY_DIR).mkdir(exist_ok=True)


def registered_env_creator(config):
    """Create a new posggym registered Driving Environment."""
    env = posggym.make(config["env_name"])
    env = FlattenObservation(env)
    return RllibMultiAgentEnv(env)

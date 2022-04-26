import random
import pathlib
import os.path as osp
from typing import List

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


def get_br_policy_mapping_fn(policy_br_id: str,
                             br_agent_id: str,
                             other_policy_ids: List[str],
                             other_policy_dist: List[float]):
    """Get policy mapping fn for BR agent."""
    def mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == br_agent_id:
            return policy_br_id
        return random.choices(
            other_policy_ids, weights=other_policy_dist, k=1
        )[0]

    return mapping_fn

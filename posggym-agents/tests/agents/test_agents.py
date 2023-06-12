import pytest

import posggym

from posggym_agents.policy import BasePolicy
from posggym_agents.agents.registration import PolicySpec

from tests.agents.spec_list import spec_list


def _has_prefix(spec: PolicySpec, env_id_prefix: str):
    return spec.id.startswith(env_id_prefix)


@pytest.mark.parametrize("spec", spec_list)
def test_policy(spec: PolicySpec, env_id_prefix: str):
    """Run reset and two test steps for each registered policy"""
    if env_id_prefix is not None and not _has_prefix(spec, env_id_prefix):
        return

    # Capture warnings
    with pytest.warns(None) as warnings:
        if spec.env_id is None:
            # policy is generic so just test on a standard env
            env = posggym.make("Driving14x14WideRoundAbout-n2-v0")
        else:
            env = posggym.make(spec.env_id)

        if spec.valid_agent_ids:
            test_agent_id = spec.valid_agent_ids[0]
        else:
            test_agent_id = 0

        test_policy = spec.make(env.model, test_agent_id)

    assert isinstance(test_policy, BasePolicy)

    for warning_msg in warnings:
        assert "autodetected dtype" not in str(warning_msg.message)

    obs = env.reset()

    if not env.observation_first:
        obs = [None] * env.n_agents

    test_policy.reset()

    for t in range(2):
        joint_action = []
        for i in range(env.n_agents):
            if i == test_agent_id:
                a = test_policy.step(obs[i])
            else:
                a = env.action_spaces[i].sample()
            joint_action.append(a)

        obs, rewards, done, info = env.step(joint_action)

    env.close()

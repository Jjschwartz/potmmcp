import copy
from itertools import product
from typing import List, Optional, Dict

import posggym
import posggym_agents

from baposgmcp.run.exp import ExpParams, PolicyParams


def get_pairwise_exp_params(env_name: str,
                            policy_params: List[List[PolicyParams]],
                            init_seed: int,
                            num_seeds: int,
                            num_episodes: int,
                            discount: float,
                            time_limit: Optional[int] = None,
                            exp_id_init: int = 0,
                            tracker_fn: Optional = None,
                            tracker_fn_kwargs: Optional[Dict] = None,
                            renderer_fn: Optional = None,
                            record_env: bool = True,
                            **kwargs) -> List[ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Assumes that the environment is symmetric.
    - Will create an experiment for every possible pairing of policies.
    """
    assert isinstance(policy_params[0], list)
    env = posggym.make(env_name)
    episode_step_limit = env.spec.max_episode_steps

    exp_params_list = []
    for i, (exp_seed, *policies) in enumerate(product(
            range(num_seeds), *policy_params
    )):
        policies = [*policies]

        exp_params = ExpParams(
            exp_id=exp_id_init+i,
            env_name=env_name,
            policy_params_list=policies,
            discount=discount,
            seed=init_seed + exp_seed,
            num_episodes=num_episodes,
            episode_step_limit=episode_step_limit,
            time_limit=time_limit,
            tracker_fn=tracker_fn,
            tracker_fn_kwargs=tracker_fn_kwargs,
            renderer_fn=renderer_fn,
            record_env=record_env,
            record_env_freq=max(1, num_episodes // 10),
            use_checkpointing=True,
        )
        exp_params_list.append(exp_params)

    return exp_params_list


def posggym_agent_entry_point(model, agent_id, kwargs):
    """Initialize a posggym agent.

    Required kwargs
    ---------------
    policy_id: str
    """
    kwargs = copy.deepcopy(kwargs)
    policy_id = kwargs.pop("policy_id")
    return posggym_agents.make(policy_id, model, agent_id, **kwargs)


def load_posggym_agent_params(policy_ids: List[str]) -> List[PolicyParams]:
    """Load posggym-agent policy params from ids."""
    return [
        PolicyParams(
            id=policy_id,
            entry_point=posggym_agent_entry_point,
            kwargs={"policy_id": policy_id},
            info=None
        )
        for policy_id in policy_ids
    ]

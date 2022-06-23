import random
from typing import Dict, Union, Any

import posggym.model as M

AgentID = Union[str, int, M.AgentID, None]

StateDist = Dict[M.State, float]
ActionDist = Dict[M.Action, float]

Policy = Any
PolicyID = Union[int, str]


def sample_action_dist(dist: ActionDist) -> M.Action:
    """Sample an action from an action distribution."""
    return random.choices(list(dist.keys()), weights=dist.values(), k=1)[0]

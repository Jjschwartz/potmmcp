from typing import Dict, Union, Tuple, Any

import posggym.model as M

AgentID = Union[str, int, M.AgentID, None]

StateDist = Dict[M.State, float]
ActionDist = Dict[M.Action, float]

Policy = Any
PolicyID = Union[int, str]
OtherAgentPolicyDist = Dict[
    M.AgentID, Tuple[Tuple[PolicyID, ...], Tuple[float, ...]]
]

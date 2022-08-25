from typing import Dict, Union, Tuple, Sequence

import posggym.model as M

AgentID = Union[str, int, M.AgentID, None]

StateDist = Dict[M.State, float]


def normalize_dist(dist: Sequence[float]) -> Tuple[float]:
    """Normalize a distribution."""
    prob_sum = sum(dist)
    return tuple(p / prob_sum for p in dist)

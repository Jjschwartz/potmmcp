from typing import Optional

from baposgmcp import parts


def get_policy_id(agent_id: Optional[parts.AgentID],
                  name: str,
                  is_symmetric: bool) -> str:
    """Get a standard format policy ID string.

    This function ensures naming consistency which helps when saving and
    loading policies, etc.
    """
    if is_symmetric:
        return f"pi_{name}"
    return f"pi_{name}_{agent_id}"


def get_agent_id_from_policy_id(policy_id: str) -> parts.AgentID:
    """Get agent id from policy id.

    Assumes pbt naming and that env is asymmetric.
    """
    tokens = policy_id.split("_")
    if len(tokens) == 2:
        return None
    return tokens[2]

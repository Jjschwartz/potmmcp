

# pylint: disable=[unused-argument]
def default_asymmetric_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Default Policy mapping function for asymmetric environments

    Assumes the policy ID naming convention used by baposgmcp.pbt.
    """
    for policy_id in episode.policy_map.keys():
        if policy_id.endswith(agent_id):
            return policy_id
    raise AssertionError


# pylint: disable=[unused-argument]
def default_symmetric_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Default Policy mapping function for asymmetric environments

    Assumes the policy ID naming convention used by baposgmcp.pbt.
    Also assumes agent id is an integer or a str representation of an integer.
    """
    policy_ids = list(episode.policy_map.keys())
    policy_ids.sort()
    return policy_ids[int(agent_id)]

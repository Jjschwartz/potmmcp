from posggym import logger

from posggym_agents import agents


def should_skip_policy_spec_for_tests(spec):
    """Get whether to skip policy spec for testing.

    We skip tests for envs that require dependencies or are otherwise
    troublesome to run frequently
    """
    if (
        spec.id.startswith("fixed-dist-random")   # requires dist arg
    ):
        logger.warn(f"Skipping tests for policy {spec.id}")
        return True
    return False


spec_list = [
    spec
    for spec in sorted(agents.registry.all(), key=lambda x: x.id)
    if not should_skip_policy_spec_for_tests(spec)
]

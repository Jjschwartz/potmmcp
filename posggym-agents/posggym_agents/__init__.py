"""Root '__init__' of the posggym package."""
# isort: skip_file
from posggym_agents.agents import make, register, registry, spec


__all__ = [
    # registration
    "make",
    "register",
    "registry",
    "spec",
]


__version__ = "0.1.2"

"""Initializes posg implemented environments.

Utilizes the OpenAI Gym env registration functionality.
"""
from posggym.envs.registration import register
from posggym.envs.registration import make  # noqa
from posggym.envs.registration import spec  # noqa
from posggym.envs.registration import registry  # noqa

from posggym.envs.grid_world import driving
from posggym.envs.grid_world import predator_prey
from posggym.envs.grid_world import pursuit_evasion


# Grid World
# -------------------------------------------

# Pursuit-Evasion
register(
    id="PursuitEvasion16x16-v0",
    entry_point=("posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv"),
    max_episode_steps=pursuit_evasion.grid.SUPPORTED_GRIDS["16x16"][1],
    kwargs={
        "grid_name": "16x16",
        "action_probs": 1.0,
        "max_obs_distance": 12,
        "normalize_reward": True,
        "use_progress_reward": True,
    },
)


# Driving
grid_fn, finite_steps = driving.grid.SUPPORTED_GRIDS["14x14WideRoundAbout"]
register(
    id="Driving14x14WideRoundAbout-n2-v0",
    entry_point="posggym.envs.grid_world.driving:DrivingEnv",
    max_episode_steps=finite_steps,
    kwargs={
        "grid": grid_fn(),
        "num_agents": 2,
        "obs_dim": (3, 1, 1),
        "obstacle_collisions": False,
        "infinite_horizon": False,
    },
)


# Predator-Prey
grid_fn, finite_steps = predator_prey.grid.SUPPORTED_GRIDS["10x10"]
register(
    id="PredatorPrey10x10-P2-p3-s2-coop-v0",
    entry_point="posggym.envs.grid_world.predator_prey:PPEnv",
    max_episode_steps=finite_steps,
    kwargs={
        "grid": grid_fn(),
        "num_predators": 2,
        "num_prey": 3,
        "cooperative": True,
        "prey_strength": 2,
        "obs_dim": 2,
    },
)

register(
    id="PredatorPrey10x10-P4-p3-s3-coop-v0",
    entry_point="posggym.envs.grid_world.predator_prey:PPEnv",
    max_episode_steps=finite_steps,
    kwargs={
        "grid": grid_fn(),
        "num_predators": 4,
        "num_prey": 3,
        "cooperative": True,
        "prey_strength": 3,
        "obs_dim": 2,
    },
)

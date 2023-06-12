import abc
import time
from typing import Sequence, Iterable

import matplotlib.pyplot as plt

import posggym
import posggym.model as M

import posggym_agents.policy as Pi


class Renderer(abc.ABC):
    """Abstract Renderer Base class."""

    @abc.abstractmethod
    def render_step(self,
                    episode_t: int,
                    env: posggym.Env,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[Pi.BasePolicy],
                    episode_end: bool) -> None:
        """Render a single environment step."""


class EpisodeRenderer(Renderer):
    """Episode Renderer.

    Calls the posggym.Env.render() function with given mode.
    """

    def __init__(self, mode: str = 'human', render_frequency: int = 1):
        self._mode = mode
        self._render_frequency = render_frequency
        self._episode_count = 0

    def render_step(self,
                    episode_t: int,
                    env: posggym.Env,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[Pi.BasePolicy],
                    episode_end: bool) -> None:
        if self._episode_count % self._render_frequency != 0:
            self._episode_count += int(episode_end)
            return

        self._episode_count += int(episode_end)

        env.render(self._mode)


class PauseRenderer(Renderer):
    """Pauses for user input after each step.

    Note, renderers are rendered in order so if you want to pause after all the
    other renderers are done for the step make sure to put the PauseRenderer
    instance at the end of the list of renderers that are passed to the
    generate_renders function, or handle calling each renderer manually.
    """

    def render_step(self,
                    episode_t: int,
                    env: posggym.Env,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[Pi.BasePolicy],
                    episode_end: bool) -> None:
        input("Press ENTER to continue.")


def generate_renders(renderers: Iterable[Renderer],
                     episode_t: int,
                     env: posggym.Env,
                     timestep: M.JointTimestep,
                     action: M.JointAction,
                     policies: Sequence[Pi.BasePolicy],
                     episode_end: bool) -> None:
    """Handle the generation of environment step renderings."""
    num_renderers = 0
    for renderer in renderers:
        renderer.render_step(
            episode_t, env, timestep, action, policies, episode_end
        )
        num_renderers += 1

    if num_renderers > 0:
        plt.show()
        # give it time to render
        time.sleep(0.1)

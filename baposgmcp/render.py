import abc
from typing import Sequence, Dict, Any

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt

import posggym
import posggym.model as M

import baposgmcp.tree as tree_lib
import baposgmcp.policy as policy_lib

# Used to map a probability to a color
prob_color_mapper = cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True),
    cmap=cm.YlOrRd     # pylint: disable=[no-member]
)


def plot_discrete_dist(ax: matplotlib.axes.Axes,
                       dist: Dict[Any, float]) -> matplotlib.axes.Axes:
    """Plot discrete dist represented by mapping from object to prob.

    Simply returns provided axes.
    """
    labels = [str(x) for x in dist]
    labels.sort()

    label_probs_map = {str(x): p for x, p in dist.items()}
    probs = [label_probs_map.get(x, 0.0) for x in labels]

    x = list(range(len(probs)))
    ax.barh(x, probs, tick_label=labels)
    ax.set_xlim(0.0, 1.0)

    return ax


class Renderer(abc.ABC):
    """Abstract Renderer Base class."""

    FIG_SIZE = (12, 20)

    @abc.abstractmethod
    def render_step(self,
                    episode_t: int,
                    env: posggym.Env,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[policy_lib.BasePolicy],
                    episode_end: bool) -> None:
        """Render a single environment step."""


class PolicyBeliefRenderer(Renderer):
    """Renders a BayesPOSGMCP policy's root belief over other agent pis."""

    def render_step(self,
                    episode_t: int,
                    env: posggym.Env,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[policy_lib.BasePolicy],
                    episode_end: bool) -> None:
        if episode_t == 0:
            return

        for policy in policies:
            if isinstance(policy, tree_lib.BAPOSGMCP):
                self._render_baposgmcp(policy)

    @staticmethod
    def _render_baposgmcp(baposgmcp: tree_lib.BAPOSGMCP) -> None:
        ncols = baposgmcp.num_agents - 1
        fig, axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            squeeze=False,
            sharey=True,
            sharex=True
        )

        pi_beliefs = tree_lib.get_other_pis_belief(baposgmcp)

        for col, (i, belief) in enumerate(pi_beliefs.items()):
            row = 0
            ax = axs[row][col]
            ax = plot_discrete_dist(ax, belief)
            ax.set_title(f"agent={i}")

        fig.suptitle(
            f"t={baposgmcp.history.t} ego_agent={baposgmcp.ego_agent}\n"
            f"Ego history={baposgmcp.history}"
        )
        fig.tight_layout()

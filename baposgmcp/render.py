from typing import Sequence

import matplotlib.pyplot as plt

import posggym
import posggym.model as M

import posgmcp.policy as policy_lib
from posgmcp.render import Renderer, plot_discrete_dist

import baposgmcp.tree as tree_lib


class PolicyBeliefRenderer(Renderer):
    """Renders a BayesPOSGMCP policy's root belief over other agent pis """

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

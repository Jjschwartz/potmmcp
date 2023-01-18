from typing import Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import posggym.model as M
from posggym.envs.grid_world.driving import DrivingEnv

import baposgmcp.policy as P
from baposgmcp.run import Renderer
from baposgmcp.tree import BAPOSGMCP


class PositionBeliefRenderer(Renderer):
    """Renders BAPOSGMCP policy's root belief over position of each agent."""

    def __init__(self, figsize: Tuple[int, int] = (9, 9)):
        self._figsize = figsize
        self._axs = None
        self._fig = None
        self._n_agents = 0
        self._grid_dims = (0, 0)

    def render_step(self,
                    episode_t: int,
                    env: DrivingEnv,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[P.BasePolicy],
                    episode_end: bool) -> None:
        if episode_t == 0:
            return

        if self._fig is None:
            num_baposgmcp = sum(
                isinstance(pi, BAPOSGMCP) for pi in policies
            )
            assert num_baposgmcp <= 1, (
                "PositionBeliefRenderer currently only support one BAPOSGMCP "
                "policy. Update me if you need support for more than 1."
            )
            if num_baposgmcp == 0:
                # incase of experiment between two non-BAPOSGMCP policies
                self._fig = 0   # placeholder, so self._fig != None
            else:
                self._fig, self._axs = plt.subplots(
                    nrows=2,
                    ncols=env.n_agents,
                    squeeze=False,
                    figsize=self._figsize
                )
                self._grid_dims = (env.model.grid.width, env.model.grid.height)
                self._n_agents = env.n_agents

                for row in range(len(self._axs)):
                    for ax in self._axs[row]:
                        # Turn off x/y axis numbering/ticks
                        ax.xaxis.set_ticks_position('none')
                        ax.yaxis.set_ticks_position('none')
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])

        for policy in policies:
            if isinstance(policy, BAPOSGMCP):
                self._render_baposgmcp(policy)

    def _render_baposgmcp(self, baposgmcp: BAPOSGMCP) -> None:
        pos_beliefs, dest_beliefs = self._get_pos_beliefs(
            baposgmcp.root.belief
        )

        for k, beliefs in enumerate([pos_beliefs, dest_beliefs]):
            for i in range(self._n_agents):
                self._axs[k][i].clear()
                self._axs[k][i].imshow(
                    beliefs[i].transpose(1, 0),
                    interpolation='bilinear',
                    origin='upper'
                )
                for (r, c), b in np.ndenumerate(beliefs[i]):
                    self._axs[k][i].text(
                        r, c,
                        '{:0.2f}'.format(b),
                        ha='center',
                        va='center',
                        bbox=dict(
                            boxstyle='round',
                            facecolor='white',
                            edgecolor='0.3')
                    )

        self._fig.canvas.draw()
        plt.pause(0.01)

    def _get_pos_beliefs(self,
                         belief: M.Belief
                         ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pos_beliefs = [
            np.zeros(self._grid_dims) for _ in range(self._n_agents)
        ]
        dest_beliefs = [
            np.zeros(self._grid_dims) for _ in range(self._n_agents)
        ]

        if belief.size() == 0:
            # BAPOSGMCP agent in terminal state
            return pos_beliefs, dest_beliefs

        for hp_state, prob in belief.get_dist().items():
            for i in range(self._n_agents):
                pos_beliefs[i][hp_state.state[i].coord] += prob
                dest_beliefs[i][hp_state.state[i].dest_coord] += prob

        # normalize
        for i in range(self._n_agents):
            pos_beliefs[i] /= pos_beliefs[i].sum()
            dest_beliefs[i] /= dest_beliefs[i].sum()

        return pos_beliefs, dest_beliefs

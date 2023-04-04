import abc
import time
from typing import Any, Dict, Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import posggym
import posggym.model as M
from matplotlib import cm

import potmmcp.policy as P
import potmmcp.tree as tree_lib


# Used to map a probability to a color
prob_color_mapper = cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True),
    cmap=cm.YlOrRd,  # pylint: disable=[no-member]
)


def plot_discrete_dist(
    ax: matplotlib.axes.Axes, dist: Dict[Any, float]
) -> matplotlib.axes.Axes:
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
    def render_step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ) -> None:
        """Render a single environment step."""


class EpisodeRenderer(Renderer):
    """Episode Renderer.

    Calls the posggym.Env.render() function with given mode.
    """

    def __init__(self, mode: str = "human", render_frequency: int = 1):
        self._mode = mode
        self._render_frequency = render_frequency
        self._episode_count = 0

    def render_step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ) -> None:
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

    def render_step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ) -> None:
        input("Press ENTER to continue.")


class PolicyBeliefRenderer(Renderer):
    """Renders a POTMMCP policy's root belief over other agent pis."""

    def __init__(self):
        self._axs = None
        self._fig = None

    def render_step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ) -> None:
        if episode_t == 0:
            return

        for policy in policies:
            if isinstance(policy, tree_lib.POTMMCP):
                self._render_potmmcp(policy)

    def _render_potmmcp(self, potmmcp: tree_lib.POTMMCP) -> None:
        if self._axs is None:
            ncols = potmmcp.num_agents - 1
            self._fig, self._axs = plt.subplots(
                nrows=1, ncols=ncols, squeeze=False, sharey=True, sharex=True
            )

        pi_beliefs = tree_lib.get_other_pis_belief(potmmcp)

        for col, (i, belief) in enumerate(pi_beliefs.items()):
            row = 0
            ax = self._axs[row][col]
            ax.clear()
            ax = plot_discrete_dist(ax, belief)
            ax.set_title(f"agent={i}")

        self._fig.suptitle(f"t={potmmcp.history.t} ego_agent={potmmcp.agent_id}")
        self._fig.tight_layout()


class SearchTreeRenderer(Renderer):
    """Renders a policies search tree."""

    def __init__(self, tree_depth: int):
        self._tree_depth = tree_depth
        self._fig = None
        self._ax = None

    def render_step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: M.JointAction,
        policies: Sequence[P.BasePolicy],
        episode_end: bool,
    ) -> None:
        if episode_t == 0:
            return

        if self._fig is None:
            num_potmmcp = sum(isinstance(pi, tree_lib.POTMMCP) for pi in policies)
            assert num_potmmcp <= 1

        for i, policy in enumerate(policies):
            if isinstance(policy, tree_lib.POTMMCP):
                self._render_tree(policy, action[i], timestep[1][i])

    def _render_tree(
        self, potmmcp: tree_lib.POTMMCP, action: M.Action, reward: float
    ) -> None:

        try:
            import networkx as nx
            from networkx.drawing.nx_agraph import graphviz_layout
        except ImportError as e:
            raise ImportError(
                "Missing dependencies 'networkx' and 'pygraphviz' for visualizing "
                "search trees. Run `pip install -e .[viz]` from potmmcp root directory "
                "to install."
            ) from e

        # if self._fig is None:
        self._fig, self._ax = plt.subplots(
            nrows=1, ncols=1, squeeze=True, figsize=(9, 9)
        )

        root_node = potmmcp.root

        graph = nx.DiGraph()
        graph.add_node(
            root_node,
            v=f"{root_node.value:.2f}",
            n=root_node.visits,
            # o=root_node.obs
        )
        self._recursively_build_tree(graph, root_node, 0)

        pos = graphviz_layout(graph, prog="dot")
        node_colors = []
        for node in graph.nodes():
            node_colors.append("#A0CBE2")

        self._ax.clear()
        nx.draw(graph, pos, self._ax, node_color=node_colors, with_labels=True)

        self._fig.suptitle(
            f"Agent {potmmcp.agent_id} t={potmmcp.history.t}\n"
            f"{action=}\n"
            f"{reward=:}"
        )
        self._fig.tight_layout()

    def _recursively_build_tree(
        self, graph, parent: tree_lib.Node, depth: int
    ):
        if len(parent.children) == 0 or depth == self._tree_depth:
            return

        for child in parent.children:
            if isinstance(child, tree_lib.ActionNode):
                graph.add_node(
                    child, v=f"{child.value:.2f}", n=child.visits, a=child.action
                )
            else:
                graph.add_node(
                    child,
                    v=f"{child.value:.2f}",
                    n=child.visits,
                    # o=child.obs
                )
            graph.add_edge(parent, child)
            self._recursively_build_tree(graph, child, depth + 1)


def generate_renders(
    renderers: Iterable[Renderer],
    episode_t: int,
    env: posggym.Env,
    timestep: M.JointTimestep,
    action: M.JointAction,
    policies: Sequence[P.BasePolicy],
    episode_end: bool,
) -> None:
    """Handle the generation of environment step renderings."""
    num_renderers = 0
    for renderer in renderers:
        renderer.render_step(episode_t, env, timestep, action, policies, episode_end)
        num_renderers += 1

    if num_renderers > 0:
        plt.show()
        # give it time to render
        time.sleep(0.1)

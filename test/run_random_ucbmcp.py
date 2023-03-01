"""Run UCB MCP with random rollout policy and random other agent."""
import argparse
import math
from typing import Optional
import logging

import posggym
import posggym.model as M
import posggym_agents

import potmmcp.run as run_lib
import potmmcp.run.stats as stats_lib
from potmmcp.meta_policy import SingleMetaPolicy
from potmmcp.policy_prior import load_posggym_agents_policy_prior
from potmmcp.run.render import EpisodeRenderer
from potmmcp.tree import POTMMCP


DISCOUNT = 0.99

UCB_KWARGS = {
    "discount": DISCOUNT,
    "c": math.sqrt(2),
    "truncated": False,
    "action_selection": "ucb",
    # "dirichlet_alpha": 0.5,  # added based on env
    "root_exploration_fraction": 0.5,
    "reinvigorator": None,  # Use default rejection sampler
    "known_bounds": None,
    "extra_particles_prop": 1.0 / 16,
    # "step_limit": # added by command line
    "epsilon": 0.01,
}


def get_ucb_planner(
    model: M.POSGModel,
    agent_id: int,
    num_sims: Optional[int],
    search_time_limit: Optional[float],
    step_limit: Optional[int] = None,
):  # noqa

    meta_policy = SingleMetaPolicy.load_possgym_agents_meta_policy(
        model, agent_id, "random-v0"
    )

    other_agent_ids = [i for i in range(model.n_agents) if i != agent_id]
    other_policy_prior = load_posggym_agents_policy_prior(
        model, agent_id, {i: {"random-v0": 1.0} for i in other_agent_ids}
    )

    ucb_planner = POTMMCP(
        model,
        agent_id,
        num_sims=num_sims,
        search_time_limit=search_time_limit,
        other_policy_prior=other_policy_prior,
        meta_policy=meta_policy,
        step_limit=model.spec.max_episode_steps if step_limit is None else step_limit,
        dirichlet_alpha=model.action_spaces[agent_id].n / 10,
        **UCB_KWARGS
    )

    return ucb_planner


def run_random_ucbmcp_planner(
    env_name: str,
    agent_id: int,
    num_sims: Optional[int],
    search_time_limit: Optional[float],
    num_episodes: int,
    episode_step_limit: Optional[int],
    seed: Optional[int],
    render: bool,
    render_mode: str,
):
    """Run random UCB MCP planner."""
    env = posggym.make(env_name)

    action_spaces = env.action_spaces

    # set random seeds
    if seed is not None:
        env.reset(seed=seed)
        for i in range(len(action_spaces)):
            action_spaces[i].seed(seed + 1 + i)

    ucb_planner = get_ucb_planner(
        env.model, agent_id, num_sims, search_time_limit, episode_step_limit
    )

    other_agent_ids = [i for i in range(env.model.n_agents) if i != agent_id]
    policies = [posggym_agents.make("random-v0", env.model, i) for i in other_agent_ids]
    policies.insert(agent_id, ucb_planner)

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    run_lib.run_episodes(
        env,
        policies,
        num_episodes,
        trackers=stats_lib.get_default_trackers(env.n_agents, DISCOUNT),
        renderers=[EpisodeRenderer()] if render else [],
        episode_step_limit=episode_step_limit,
        logger=logger
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("env_name", type=str, help="Name of environment to run")
    parser.add_argument(
        "--agent_id",
        type=int,
        default=0,
        help="The ID of agent to run UCB MCP planner for",
    )
    parser.add_argument(
        "--num_sims", type=int, default=None, help="Number of simulations per step"
    )
    parser.add_argument(
        "--search_time_limit",
        type=float,
        default=None,
        help="Search time limit per step",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1, help="The number of episodes to run"
    )
    parser.add_argument(
        "--episode_step_limit",
        type=int,
        default=None,
        help="Max number of steps to run each epsiode for",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random Seed")
    parser.add_argument(
        "--render", action="store_true", help="Render environment steps"
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        help="Mode to use for renderering, if rendering",
    )
    args = parser.parse_args()
    run_random_ucbmcp_planner(**vars(args))

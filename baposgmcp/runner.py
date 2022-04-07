import time
import random
import logging
from typing import Sequence, Optional, Iterable, NamedTuple
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

import posggym
import posggym.model as M

import baposgmcp.stats as stats_lib
import baposgmcp.render as render_lib
from baposgmcp.policy import BasePolicy


LINE_BREAK = "-"*60
MAJOR_LINE_BREAK = "="*60


class EpisodeLoopStep(NamedTuple):
    """Output for a single episode step."""
    env: posggym.Env
    timestep: M.JointTimestep
    actions: Optional[M.JointAction]
    policies: Sequence[BasePolicy]
    done: bool


class RunConfig(NamedTuple):
    """Configuration options for running simulations."""
    seed: Optional[int] = None
    num_episodes: int = 100
    episode_step_limit: int = 20
    time_limit: Optional[int] = None


def get_run_args(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    """Get command line arguments for running episodes."""
    if parser is None:
        parser = ArgumentParser(
            conflict_handler='resolve',
            formatter_class=ArgumentDefaultsHelpFormatter
        )

    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="The number of episodes to run"
    )
    parser.add_argument(
        "--episode_step_limit", type=int, default=20,
        help="Max number of steps per episode"
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Time limit (s) for running all simulations"
    )
    parser.add_argument(
        "--render_mode", type=str, default=None,
        help="Render mode for rendering episodes"
    )
    parser.add_argument(
        "--pause_each_step", action="store_true",
        help="Pause after each step is executed"
    )
    parser.add_argument(
        "--log_level", type=str, default='INFO',
        help="Logging level"
    )

    return parser


def run_episode_loop(env: posggym.Env,
                     policies: Sequence[BasePolicy],
                     step_limit: int,
                     ) -> Iterable[EpisodeLoopStep]:
    """Run policies in environment."""
    assert len(policies) == env.n_agents, (
        f"{len(policies)} policies supplied for environment with "
        f"{env.n_agents} agents."
    )

    joint_obs = env.reset()
    joint_timestep = (    # type: ignore
        joint_obs, tuple(0.0 for _ in range(env.n_agents)), False, {}
    )
    init_action = None
    yield EpisodeLoopStep(env, joint_timestep, init_action, policies, False)

    episode_end = False
    steps = 0
    while not episode_end:
        agent_actions = []
        for i in range(env.n_agents):
            agent_actions.append(policies[i].step(joint_obs[i]))

        joint_action = tuple(agent_actions)
        joint_timestep = env.step(joint_action)

        steps += 1
        joint_obs = joint_timestep[0]
        episode_end = joint_timestep[2] or steps >= step_limit

        yield EpisodeLoopStep(
            env, joint_timestep, joint_action, policies, episode_end
        )


# pylint: disable=[unused-argument]
def run_sims(env: posggym.Env,
             policies: Sequence[BasePolicy],
             trackers: Iterable[stats_lib.Tracker],
             renderers: Iterable[render_lib.Renderer],
             run_config: RunConfig,
             logger: Optional[logging.Logger] = None
             ) -> stats_lib.AgentStatisticsMap:
    """Run Episode simulations for given env and policies."""
    logger = logging.getLogger() if logger is None else logger

    logger.info(
        "%s\nRunning %d episodes with Time Limit = %s s\n%s",
        MAJOR_LINE_BREAK,
        run_config.num_episodes,
        str(run_config.time_limit),
        MAJOR_LINE_BREAK
    )

    if run_config.seed is not None:
        random.seed(run_config.seed)
        np.random.seed(run_config.seed)

    episode_num = 0
    progress_display_freq = max(1, run_config.num_episodes // 10)
    time_limit_reached = False
    run_start_time = time.time()

    for tracker in trackers:
        tracker.reset()

    while episode_num < run_config.num_episodes and not time_limit_reached:
        logger.log(
            logging.INFO - 1,
            "%s\nEpisode %d Start\n%s",
            MAJOR_LINE_BREAK,
            episode_num,
            MAJOR_LINE_BREAK
        )

        for tracker in trackers:
            tracker.reset_episode()

        for policy in policies:
            policy.reset()

        timestep_sequence = run_episode_loop(
            env, policies, run_config.episode_step_limit
        )
        for t, loop_step in enumerate(timestep_sequence):
            for tracker in trackers:
                tracker.step(t, *loop_step)
            render_lib.generate_renders(renderers, t, *loop_step)

        episode_statistics = stats_lib.generate_episode_statistics(trackers)
        logger.log(
            logging.INFO - 1,
            "%s\nEpisode %d Complete\n%s",
            LINE_BREAK,
            episode_num,
            stats_lib.format_as_table(episode_statistics)
        )

        if (episode_num + 1) % progress_display_freq == 0:
            logger.info(
                "Episode %d / %d complete",
                episode_num + 1,
                run_config.num_episodes
            )

        episode_num += 1

        if (
            run_config.time_limit is not None
            and time.time()-run_start_time > run_config.time_limit
        ):
            time_limit_reached = True
            logger.info(
                "%s\nTime limit of %d s reached after %d episodes",
                MAJOR_LINE_BREAK,
                run_config.time_limit,
                episode_num
            )

    statistics = stats_lib.generate_statistics(trackers)

    logger.info(
        "%s\nSimulations Complete\n%s\n%s",
        MAJOR_LINE_BREAK,
        stats_lib.format_as_table(statistics),
        MAJOR_LINE_BREAK
    )

    return statistics


def run_sims_from_args(env: posggym.Env,
                       policies: Sequence[BasePolicy],
                       args) -> stats_lib.AgentStatisticsMap:
    """Run episode sims for given env and policies.

    This function handles generating trackers, renderers, loggers from an
    arguments object.
    """
    trackers = stats_lib.get_default_trackers(policies)
    renderers = render_lib.get_renderers(
        render_mode=args.render_mode,
        pause_each_step=args.pause_each_step,
        show_pi=args.show_pi,
        show_belief=args.show_belief,
        show_tree=args.show_tree
    )

    run_config = RunConfig(
        seed=args.seed,
        num_episodes=args.num_episodes,
        episode_step_limit=args.episode_step_limit,
        time_limit=args.time_limit
    )

    logging.basicConfig(level=args.log_level, format='%(message)s')
    logger = logging.getLogger()

    return run_sims(env, policies, trackers, renderers, run_config, logger)

import time
import logging
from typing import Sequence, Optional, Iterable, NamedTuple

import posggym
import posggym.model as M

from posggym_agents.policy import BasePolicy
import posggym_agents.exp.stats as stats_lib
import posggym_agents.exp.render as render_lib
import posggym_agents.exp.writer as writer_lib


LINE_BREAK = "-"*60
MAJOR_LINE_BREAK = "="*60


class EpisodeLoopStep(NamedTuple):
    """Output for a single episode step."""
    env: posggym.Env
    timestep: M.JointTimestep
    actions: Optional[M.JointAction]
    policies: Sequence[BasePolicy]
    done: bool


def run_episode_loop(env: posggym.Env,
                     policies: Sequence[BasePolicy],
                     ) -> Iterable[EpisodeLoopStep]:
    """Run policies in environment."""
    assert len(policies) == env.n_agents, (
        f"{len(policies)} policies supplied for environment with "
        f"{env.n_agents} agents."
    )

    joint_obs = env.reset()

    if not env.observation_first:
        joint_obs = [None] * env.n_agents

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
        episode_end = joint_timestep[2]

        yield EpisodeLoopStep(
            env, joint_timestep, joint_action, policies, episode_end
        )


def run_episode(env: posggym.Env,
                policies: Sequence[BasePolicy],
                num_episodes: int,
                trackers: Iterable[stats_lib.Tracker],
                renderers: Iterable[render_lib.Renderer],
                time_limit: Optional[int] = None,
                logger: Optional[logging.Logger] = None,
                writer: Optional[writer_lib.Writer] = None
                ) -> stats_lib.AgentStatisticsMap:
    """Run Episode simulations for given env and policies."""
    logger = logging.getLogger() if logger is None else logger
    writer = writer_lib.NullWriter() if writer is None else writer

    logger.info(
        "%s\nRunning %d episodes with Time Limit = %s s\n%s",
        MAJOR_LINE_BREAK,
        num_episodes,
        str(time_limit),
        MAJOR_LINE_BREAK
    )

    episode_num = 0
    progress_display_freq = max(1, num_episodes // 10)
    time_limit_reached = False
    run_start_time = time.time()

    for tracker in trackers:
        tracker.reset()

    while episode_num < num_episodes and not time_limit_reached:
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

        timestep_sequence = run_episode_loop(env, policies)
        for t, loop_step in enumerate(timestep_sequence):
            for tracker in trackers:
                tracker.step(t, *loop_step)
            render_lib.generate_renders(renderers, t, *loop_step)

        episode_statistics = stats_lib.generate_episode_statistics(trackers)
        writer.write_episode(episode_statistics)

        logger.log(
            logging.INFO - 1,
            "%s\nEpisode %d Complete\n%s",
            LINE_BREAK,
            episode_num,
            writer_lib.format_as_table(episode_statistics)
        )

        if (episode_num + 1) % progress_display_freq == 0:
            logger.info(
                "Episode %d / %d complete", episode_num + 1, num_episodes
            )

        episode_num += 1

        if time_limit is not None and time.time()-run_start_time > time_limit:
            time_limit_reached = True
            logger.info(
                "%s\nTime limit of %d s reached after %d episodes",
                MAJOR_LINE_BREAK,
                time_limit,
                episode_num
            )

    statistics = stats_lib.generate_statistics(trackers)

    logger.info(
        "%s\nSimulations Complete\n%s\n%s",
        MAJOR_LINE_BREAK,
        writer_lib.format_as_table(statistics),
        MAJOR_LINE_BREAK
    )

    return statistics

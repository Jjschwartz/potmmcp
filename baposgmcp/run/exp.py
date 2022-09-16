"""Functions and data structures for running experiments."""
import os
import copy
import time
import random
import logging
import pathlib
import argparse
import tempfile
from pprint import pformat
import multiprocessing as mp
from datetime import datetime
from typing import (
    List, Optional, Dict, Any, NamedTuple, Callable, Sequence, Set, Tuple
)

import ray
import numpy as np

import posggym
from posggym import wrappers

import baposgmcp.policy as P
from baposgmcp.run import runner
import baposgmcp.run.stats as stats_lib
import baposgmcp.run.render as render_lib
import baposgmcp.run.writer as writer_lib
from baposgmcp.config import BASE_RESULTS_DIR


LINE_BREAK = "-"*60
EXP_ARG_FILE_NAME = "exp_args.json"


# A global lock used for controlling when processes print to stdout
# This helps keep top level stdout readable
LOCK = mp.Lock()


def _init_lock(lck):
    # pylint: disable=[global-statement]
    global LOCK
    LOCK = lck


class PolicyParams(NamedTuple):
    """Params for a policy in a single experiment run.

    `entry_point` should be a function which takes arguments
    [model: posggym.POSGModel, agent_id: M.AgentID, kwargs]
    and return a policy.

    `info` is an optional dictionary whose contents will be saved to the
    results file. It can be used to add additional information alongside the
    policy, such as additional identifying info like the env trained on,
    nesting level, population ID, etc.

    add "logger" to kwargs with None if you want experiment logger to be added
    to kwargs
    """
    id: str
    kwargs: Dict[str, Any]
    entry_point: Callable[..., P.BasePolicy]
    info: Optional[Dict[str, Any]] = None


class ExpParams(NamedTuple):
    """Params for a single experiment run."""
    exp_id: int
    env_name: str
    policy_params_list: List[PolicyParams]
    # Used for tracking discounted return
    discount: float
    seed: int
    num_episodes: int
    episode_step_limit: Optional[int] = None
    time_limit: Optional[int] = None
    tracker_fn: Optional[Callable[[], Sequence[stats_lib.Tracker]]] = None
    renderer_fn: Optional[Sequence[render_lib.Renderer]] = None
    record_env: bool = False
    # If None then uses the default cubic frequency
    record_env_freq: Optional[int] = None
    # Whether to write results to file after each episode rather than just
    # at the end of the experiment
    use_checkpointing: bool = True
    stream_log_level: int = logging.INFO
    file_log_level: int = logging.DEBUG


def get_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default experiment args."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--n_procs", type=int, default=1,
        help="Number of processors/experiments to run in parallel."
    )
    parser.add_argument(
        "--log_level", type=int, default=21,
        help="Experiment log level."
    )
    parser.add_argument(
        "--using_ray", action="store_true",
        help=(
            "Whether experiment is using ray. This should be set for all "
            "experiments that use ray."
        )
    )
    parser.add_argument(
        "--root_save_dir", type=str, default=None,
        help=(
            "Optional directory to save results in. If supplied then it must "
            "be an existing directory. If None uses default "
            "~/baposgmcp_results/<env_name>/results/ dir as root results dir."
        )
    )
    return parser


def make_exp_result_dir(exp_name: str,
                        env_name: str,
                        root_save_dir: Optional[str] = None) -> str:
    """Make a directory for experiment results."""
    time_str = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    if root_save_dir is None:
        root_save_dir = os.path.join(BASE_RESULTS_DIR, env_name, "results")
    pathlib.Path(root_save_dir).mkdir(parents=True, exist_ok=True)
    result_dir = tempfile.mkdtemp(
        prefix=f"{exp_name}_{time_str}", dir=root_save_dir
    )
    return result_dir


def _log_exp_start(params: ExpParams,
                   result_dir: str,
                   logger: logging.Logger):
    LOCK.acquire()
    try:
        logger.info(LINE_BREAK)
        logger.info("Running with:")
        logger.info(pformat(params))
        logger.info(f"Result dir = {result_dir}")
        logger.info(LINE_BREAK)
    finally:
        LOCK.release()


def _log_exp_end(params: ExpParams,
                 result_dir: str,
                 logger: logging.Logger,
                 exp_time: float):
    LOCK.acquire()
    try:
        logger.info(LINE_BREAK)
        logger.info(f"Finished exp num {params.exp_id}")
        logger.info(f"Result dir = {result_dir}")
        logger.info(f"Experiment Run time {exp_time:.2f} seconds")
        logger.info(LINE_BREAK)
    finally:
        LOCK.release()


def get_exp_run_logger(exp_id: int,
                       result_dir: str,
                       stream_log_level: int = logging.INFO,
                       file_log_level: int = logging.DEBUG) -> logging.Logger:
    """Get the logger for a single experiment run."""
    logger_name = f"exp_{exp_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(stream_log_level, file_log_level))

    fname = f"exp_{exp_id}.log"
    log_file = os.path.join(result_dir, fname)
    file_formatter = logging.Formatter(
        # [Day-Month Hour-Minute-Second] exp_x Message
        '[%(asctime)s] %(levelname)s %(message)s', '%d-%m %H:%M:%S'
    )

    filehandler = logging.FileHandler(log_file)
    filehandler.setFormatter(file_formatter)
    filehandler.setLevel(file_log_level)

    stream_formatter = logging.Formatter(
        # [Day-Month Hour-Minute-Second] exp_x Message
        '[%(asctime)s] %(name)s %(message)s', '%d-%m %H:%M:%S'
    )
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(stream_formatter)
    streamhandler.setLevel(stream_log_level)

    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.propagate = False

    return logger


def _get_param_statistics(params: ExpParams
                          ) -> stats_lib.AgentStatisticsMap:
    num_agents = len(params.policy_params_list)

    def _add_dict(stat_dict, param_dict):
        for k, v in param_dict.items():
            if v is None:
                # Need to do this so Pandas imports 'None' instead of NaN
                v = 'None'
            stat_dict[k] = v

    stats = {}
    policy_headers: Set[str] = set()
    for i in range(num_agents):
        stats[i] = {
            "exp_id": params.exp_id,
            "agent_id": i,
            "env_name": params.env_name,
            "exp_seed": params.seed,
            "num_episodes": params.num_episodes,
            "time_limit": params.time_limit if params.time_limit else "None",
            "episode_step_limit": (
                params.episode_step_limit
                if params.episode_step_limit else "None"
            )
        }
        pi_params = params.policy_params_list[i]
        stats[i]["policy_id"] = pi_params.id
        _add_dict(stats[i], pi_params.kwargs)
        policy_headers.update(pi_params.kwargs)

        if pi_params.info:
            _add_dict(stats[i], pi_params.info)
            policy_headers.update(pi_params.info)

    for i in range(num_agents):
        for header in policy_headers:
            if header not in stats[i]:
                stats[i][header] = 'None'

    return stats


def _get_linear_episode_trigger(freq: int) -> Callable[[int], bool]:
    return lambda t: t % freq == 0


def _make_env(params: ExpParams, result_dir: str) -> posggym.Env:
    env = posggym.make(params.env_name, **{"seed": params.seed})
    if params.record_env:
        video_folder = os.path.join(result_dir, f"exp_{params.exp_id}_video")
        if params.record_env_freq:
            episode_trigger = _get_linear_episode_trigger(
                params.record_env_freq
            )
        else:
            episode_trigger = None
        env = wrappers.RecordVideo(env, video_folder, episode_trigger)
    return env


def run_single_experiment(args: Tuple[ExpParams, str]):
    """Run a single experiment and write results to a file."""
    params, result_dir = args
    exp_start_time = time.time()

    exp_logger = get_exp_run_logger(
        params.exp_id,
        result_dir,
        params.stream_log_level,
        params.file_log_level
    )
    _log_exp_start(params, result_dir, exp_logger)

    seed = params.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = _make_env(params, result_dir)

    policies: List[P.BasePolicy] = []
    for i, pi_params in enumerate(params.policy_params_list):
        kwargs = copy.copy(pi_params.kwargs)
        if "logger" in kwargs:
            kwargs["logger"] = exp_logger
        pi = pi_params.entry_point(env.model, i, pi_params.kwargs)
        policies.append(pi)

    if params.tracker_fn:
        trackers = params.tracker_fn()
    else:
        trackers = stats_lib.get_default_trackers(
            env.n_agents, params.discount
        )

    renderers = params.renderer_fn() if params.renderer_fn else []
    writer = writer_lib.ExperimentWriter(
        params.exp_id, result_dir, _get_param_statistics(params)
    )

    try:
        statistics = runner.run_episodes(
            env,
            policies,
            params.num_episodes,
            trackers,
            renderers,
            time_limit=params.time_limit,
            episode_step_limit=params.episode_step_limit,
            logger=exp_logger,
            writer=writer,
            use_checkpointing=params.use_checkpointing
        )
        writer.write(statistics)

    except Exception as ex:
        exp_logger.exception("Exception occured: %s", str(ex))
        exp_logger.error(pformat(locals()))
        raise ex
    finally:
        _log_exp_end(
            params, result_dir, exp_logger, time.time() - exp_start_time
        )


def run_experiments(exp_name: str,
                    exp_params_list: List[ExpParams],
                    exp_log_level: int = logging.INFO+1,
                    n_procs: Optional[int] = None,
                    using_ray: bool = False,
                    exp_args: Optional[Dict] = None,
                    root_save_dir: Optional[str] = None) -> str:
    """Run series of experiments.

    If exp_args is not None then will write to file in the result dir.
    """
    exp_start_time = time.time()
    logging.basicConfig(
        level=exp_log_level,
        # [Day-Month Hour-Minute-Second] Message
        format='[%(asctime)s] %(message)s', datefmt='%d-%m %H:%M:%S'
    )

    num_exps = len(exp_params_list)
    logging.log(exp_log_level, "Running %d experiments", num_exps)

    result_dir = make_exp_result_dir(
        exp_name, exp_params_list[0].env_name, root_save_dir
    )
    logging.log(exp_log_level, "Saving results to dir=%s", result_dir)

    if exp_args:
        writer_lib.write_dict(
            exp_args,
            os.path.join(result_dir, EXP_ARG_FILE_NAME)
        )

    if n_procs is None:
        n_procs = os.cpu_count()
    logging.log(exp_log_level, "Running %d processes", n_procs)

    mp_lock = mp.Lock()

    def _initializer(init_args):
        proc_lock = init_args
        _init_lock(proc_lock)
        if using_ray:
            # limit ray to using only a single CPU per experiment process
            logging.log(exp_log_level, "Initializing ray")
            ray.init(num_cpus=1, include_dashboard=False)

    if n_procs == 1:
        _initializer(mp_lock)
        for params in exp_params_list:
            run_single_experiment((params, result_dir))
    else:
        args_list = [(params, result_dir) for params in exp_params_list]
        with mp.Pool(
                n_procs, initializer=_initializer, initargs=(mp_lock,)
        ) as p:
            p.map(run_single_experiment, args_list, 1)

    logging.log(exp_log_level, "Compiling results")
    writer_lib.compile_results(result_dir)

    logging.log(
        exp_log_level,
        "Experiment Run time %.2f seconds",
        time.time() - exp_start_time
    )

    return result_dir

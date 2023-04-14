"""Functions and data structures for running experiments."""
import copy
import logging
import multiprocessing as mp
import os
import pathlib
import random
import tempfile
import time
from datetime import datetime
from pprint import pformat
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import posggym
from posggym import wrappers

import potmmcp.policy as P
import potmmcp.run.render as render_lib
import potmmcp.run.stats as stats_lib
import potmmcp.run.writer as writer_lib
from potmmcp.config import BASE_RESULTS_DIR
from potmmcp.run import runner


LINE_BREAK = "-" * 60
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

    def __str__(self):
        kwarg_strs = []
        for k, v in self.kwargs.items():
            if not isinstance(v, (tuple, list, dict, set)):
                kwarg_strs.append(f"{k}: {v}")
            else:
                kwarg_strs.append(f"{k}: ...")
        kwarg_str = ",\n    ".join(kwarg_strs)

        return (
            f"{self.__class__.__name__}(\n"
            f"  id={self.id}\n"
            "  kwargs={\n"
            f"    {kwarg_str}\n"
            f"  entry_point={self.entry_point}\n"
            f"  info={self.info}\n"
            ")"
        )

    def __repr__(self):
        return self.__str__()


class ExpParams(NamedTuple):
    """Params for a single experiment run."""

    exp_id: int
    env_id: str
    policy_params_list: List[PolicyParams]
    # Used for tracking discounted return
    discount: float
    seed: int
    num_episodes: int
    episode_step_limit: Optional[int] = None
    time_limit: Optional[int] = None
    tracker_fn: Optional[Callable[[Dict[str, Any]], Sequence[stats_lib.Tracker]]] = None
    tracker_fn_kwargs: Optional[Dict[str, Any]] = None
    renderer_fn: Optional[Callable[[], Sequence[render_lib.Renderer]]] = None
    record_env: bool = False
    # If None then uses the default cubic frequency
    record_env_freq: Optional[int] = None
    # Whether to write results to file after each episode rather than just
    # at the end of the experiment
    use_checkpointing: bool = True
    stream_log_level: int = logging.INFO
    file_log_level: int = logging.DEBUG

    def __str__(self):
        pi_params_str = "\n".join([str(pi) for pi in self.policy_params_list])

        arg_strs = [
            f"{k}={v}" for k, v in self._asdict().items() if k != "policy_params_list"
        ]
        arg_strs.append(f"policy_params_list=[\n{pi_params_str}\n  ]")

        arg_str = "\n  ".join(arg_strs)
        return (
            f"{self.__class__.__name__}(\n"
            f"  {arg_str}\n"
            ")"
        )


def make_exp_result_dir(
    exp_name: str, env_id: str, root_save_dir: Optional[str] = None
) -> str:
    """Make a directory for experiment results."""
    time_str = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    if root_save_dir is None:
        root_save_dir = os.path.join(BASE_RESULTS_DIR, env_id)
    pathlib.Path(root_save_dir).mkdir(parents=True, exist_ok=True)
    result_dir = tempfile.mkdtemp(prefix=f"{exp_name}_{time_str}", dir=root_save_dir)
    return result_dir


def _log_exp_start(params: ExpParams, result_dir: str, logger: logging.Logger):
    LOCK.acquire()
    try:
        logger.info(LINE_BREAK)
        logger.info("Running with:")
        # use as dict so we pformat includes newlines
        logger.info(pformat(params._asdict()))
        logger.info(f"Result dir = {result_dir}")
        logger.info(LINE_BREAK)
    finally:
        LOCK.release()


def _log_exp_end(
    params: ExpParams, result_dir: str, logger: logging.Logger, exp_time: float
):
    LOCK.acquire()
    try:
        logger.info(LINE_BREAK)
        logger.info(f"Finished exp num {params.exp_id}")
        logger.info(f"Result dir = {result_dir}")
        logger.info(f"Experiment Run time {exp_time:.2f} seconds")
        logger.info(LINE_BREAK)
    finally:
        LOCK.release()


def get_exp_run_logger(
    exp_id: int,
    result_dir: str,
    stream_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
) -> logging.Logger:
    """Get the logger for a single experiment run."""
    logger_name = f"exp_{exp_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(stream_log_level, file_log_level))

    fname = f"exp_{exp_id}.log"
    log_file = os.path.join(result_dir, fname)
    file_formatter = logging.Formatter(
        # [Day-Month Hour-Minute-Second] exp_x Message
        "[%(asctime)s] %(levelname)s %(message)s",
        "%d-%m %H:%M:%S",
    )

    filehandler = logging.FileHandler(log_file)
    filehandler.setFormatter(file_formatter)
    filehandler.setLevel(file_log_level)

    stream_formatter = logging.Formatter(
        # [Day-Month Hour-Minute-Second] exp_x Message
        "[%(asctime)s] %(name)s %(message)s",
        "%d-%m %H:%M:%S",
    )
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(stream_formatter)
    streamhandler.setLevel(stream_log_level)

    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.propagate = False

    return logger


def _get_param_statistics(params: ExpParams) -> stats_lib.AgentStatisticsMap:
    num_agents = len(params.policy_params_list)

    def _add_dict(stat_dict, param_dict):
        for k, v in param_dict.items():
            if v is None:
                # Need to do this so Pandas imports 'None' instead of NaN
                v = "None"
            stat_dict[k] = v

    stats = {}
    policy_headers: Set[str] = set()
    for i in range(num_agents):
        stats[i] = {
            "exp_id": params.exp_id,
            "agent_id": i,
            "env_id": params.env_id,
            "exp_seed": params.seed,
            "num_episodes": params.num_episodes,
            "time_limit": params.time_limit if params.time_limit else "None",
            "episode_step_limit": (
                params.episode_step_limit if params.episode_step_limit else "None"
            ),
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
                stats[i][header] = "None"

    return stats


def _get_linear_episode_trigger(freq: int) -> Callable[[int], bool]:
    return lambda t: t % freq == 0


def _make_env(params: ExpParams, result_dir: str) -> posggym.Env:
    env = posggym.make(params.env_id, **{"seed": params.seed})
    if params.record_env:
        video_folder = os.path.join(result_dir, f"exp_{params.exp_id}_video")
        if params.record_env_freq:
            episode_trigger = _get_linear_episode_trigger(params.record_env_freq)
        else:
            episode_trigger = None
        env = wrappers.RecordVideo(env, video_folder, episode_trigger)
    return env


def run_single_experiment(args: Tuple[ExpParams, str]):
    """Run a single experiment and write results to a file."""
    params, result_dir = args
    exp_start_time = time.time()

    exp_logger = get_exp_run_logger(
        params.exp_id, result_dir, params.stream_log_level, params.file_log_level
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
        tracker_fn_kwargs = params.tracker_fn_kwargs
        if not tracker_fn_kwargs:
            tracker_fn_kwargs = {}
        trackers = params.tracker_fn(tracker_fn_kwargs)
    else:
        trackers = stats_lib.get_default_trackers(env.n_agents, params.discount)

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
            use_checkpointing=params.use_checkpointing,
        )
        writer.write(statistics)

    except Exception as ex:
        exp_logger.exception("Exception occured: %s", str(ex))
        exp_logger.error(pformat(locals()))
        raise ex
    finally:
        _log_exp_end(params, result_dir, exp_logger, time.time() - exp_start_time)


def run_experiments(
    exp_name: str,
    exp_params_list: List[ExpParams],
    exp_log_level: int = logging.INFO + 1,
    n_procs: Optional[int] = None,
    exp_args: Optional[Dict] = None,
    root_save_dir: Optional[str] = None,
    run_exp_id: Optional[int] = None,
) -> str:
    """Run series of experiments.

    If exp_args is not None then will write to file in the result dir.
    """
    exp_start_time = time.time()
    logging.basicConfig(
        level=exp_log_level,
        # [Day-Month Hour-Minute-Second] Message
        format="[%(asctime)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )

    num_exps = len(exp_params_list)
    if run_exp_id is not None:
        logging.log(
            exp_log_level,
            "Running experiment %d of %d experiments",
            run_exp_id,
            num_exps,
        )
        exp_params_list = [exp_params_list[run_exp_id]]
    else:
        logging.log(exp_log_level, "Running %d experiments", num_exps)

    result_dir = make_exp_result_dir(exp_name, exp_params_list[0].env_id, root_save_dir)
    logging.log(exp_log_level, "Saving results to dir=%s", result_dir)

    if exp_args:
        writer_lib.write_dict(exp_args, os.path.join(result_dir, EXP_ARG_FILE_NAME))

    if n_procs is None:
        n_procs = os.cpu_count()

    mp_lock = mp.Lock()

    def _initializer(init_args):
        proc_lock = init_args
        _init_lock(proc_lock)

    if n_procs == 1 or run_exp_id is not None or len(exp_params_list) <= 1:
        logging.log(exp_log_level, "Running on single process")
        _initializer(mp_lock)
        for params in exp_params_list:
            run_single_experiment((params, result_dir))
    else:
        logging.log(exp_log_level, "Running %d processes", n_procs)
        args_list = [(params, result_dir) for params in exp_params_list]
        with mp.Pool(n_procs, initializer=_initializer, initargs=(mp_lock,)) as p:
            p.map(run_single_experiment, args_list, 1)

    logging.log(exp_log_level, "Compiling results")
    writer_lib.compile_results(result_dir)

    logging.log(
        exp_log_level, "Experiment Run time %.2f seconds", time.time() - exp_start_time
    )

    return result_dir

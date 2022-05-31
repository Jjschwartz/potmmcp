"""Functions and data structures for running experiments."""
import os
import copy
import json
import time
import random
import logging
from pprint import pformat
import multiprocessing as mp
from typing import (
    List, Optional, Dict, Any, NamedTuple, Callable, Sequence, Set, Tuple
)

import ray

import numpy as np

import posggym

from baposgmcp import runner
import baposgmcp.stats as stats_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as policy_lib


LINE_BREAK = "-"*60
EXP_ARG_FILE_NAME = "exp_args.json"


class PolicyParams(NamedTuple):
    """Params for a policy in a single experiment run.

    `init` should be a function which takes arguments
    [model: posggym.POSGModel, agent_id: M.AgentID, gamma: float, kwargs]
    and return a policy. The experiment logger will be added to the kwargs to
    handle logging to the correct log file.

    `info` is an optional dictionary whose contents will be saved to the
    results file. It can be used to add additional information alongside the
    policy, such as additional identifying info like the env trained on,
    nesting level, population ID, etc.
    """
    name: str
    gamma: float
    kwargs: Dict[str, Any]
    init: Callable[..., policy_lib.BasePolicy]
    info: Optional[Dict[str, Any]] = None


class ExpParams(NamedTuple):
    """Params for a single experiment run."""
    exp_id: int
    env_name: str
    policy_params_list: List[PolicyParams]
    run_config: runner.RunConfig
    tracker_fn: Optional[
        Callable[
            [List[policy_lib.BasePolicy], Dict[str, Any]],
            Sequence[stats_lib.Tracker]
        ]
    ] = None
    tracker_kwargs: Optional[Dict[str, Any]] = None
    renderer_fn: Optional[
        Callable[[Dict[str, Any]], Sequence[render_lib.Renderer]]
    ] = None
    renderer_kwargs: Optional[Dict[str, Any]] = None
    stream_log_level: int = logging.INFO
    file_log_level: int = logging.DEBUG
    setup_fn: Optional[Callable] = None
    cleanup_fn: Optional[Callable] = None


# A global lock used for controlling when processes print to stdout
# This helps keep top level stdout readable
LOCK = mp.Lock()


def _init_lock(lck):
    # pylint: disable=[global-statement]
    global LOCK
    LOCK = lck


def _log_exp_start(params: ExpParams,
                   result_dir: str,
                   logger: logging.Logger):
    LOCK.acquire()
    try:
        logger.info(LINE_BREAK)
        logger.info(f"Running exp num {params.exp_id} with:")
        logger.info(f"Env = {params.env_name}")

        for i, pi_params in enumerate(params.policy_params_list):
            logger.info(f"Agent = {i} Policy class = {pi_params.name}")
            logger.info("Policy kwargs:")
            logger.info(pformat(pi_params.kwargs))
            if pi_params.info:
                logger.info("Policy info:")
                logger.info(pformat(pi_params.info))

        logger.info("Run Config:")
        logger.info(pformat(params.run_config))
        logger.info(f"Result dir = {result_dir}")
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
        '%(asctime)s %(levelname)s %(message)s', '%H:%M:%S'
    )

    filehandler = logging.FileHandler(log_file)
    filehandler.setFormatter(file_formatter)
    filehandler.setLevel(file_log_level)

    stream_formatter = logging.Formatter('%(name)s - %(message)s')
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
        }
        # pylint: disable=[protected-access]
        _add_dict(stats[i], params.run_config._asdict())

        pi_params = params.policy_params_list[i]
        stats[i]["policy_name"] = pi_params.name
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


def _get_exp_trackers(params: ExpParams,
                      policies: List[policy_lib.BasePolicy]
                      ) -> Sequence[stats_lib.Tracker]:
    if params.tracker_fn:
        tracker_kwargs = params.tracker_kwargs if params.tracker_kwargs else {}
        trackers = params.tracker_fn(
            policies, **tracker_kwargs
        )
    else:
        trackers = stats_lib.get_default_trackers(policies)
    return trackers


def _get_exp_renderers(params: ExpParams) -> Sequence[render_lib.Renderer]:
    if params.renderer_fn:
        renderer_kwargs = {}
        if params.renderer_kwargs:
            renderer_kwargs = params.renderer_kwargs
        renderers = params.renderer_fn(**renderer_kwargs)
    else:
        renderers = []
    return renderers


def run_single_experiment(args: Tuple[ExpParams, str]) -> str:
    """Run a single experiment and write results to a file."""
    params, result_dir = args

    if params.setup_fn is not None:
        params.setup_fn(params)

    exp_logger = get_exp_run_logger(
        params.exp_id,
        result_dir,
        params.stream_log_level,
        params.file_log_level
    )
    _log_exp_start(params, result_dir, exp_logger)

    seed = params.run_config.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = posggym.make(params.env_name)

    policies: List[policy_lib.BasePolicy] = []
    for i, pi_params in enumerate(params.policy_params_list):
        kwargs = copy.copy(pi_params.kwargs)
        kwargs["logger"] = exp_logger
        pi = pi_params.init(env.model, i, pi_params.gamma, **pi_params.kwargs)
        policies.append(pi)

    trackers = _get_exp_trackers(params, policies)
    renderers = _get_exp_renderers(params)

    try:
        statistics = runner.run_sims(
            env,
            policies,
            trackers,
            renderers,
            params.run_config,
            logger=exp_logger
        )
        param_stats = _get_param_statistics(params)
        statistics = stats_lib.combine_statistics([statistics, param_stats])

        fname = os.path.join(result_dir, f"exp_{params.exp_id}.csv")
        csv_writer = stats_lib.CSVWriter(filepath=fname)
        csv_writer.write(statistics)
        csv_writer.close()

    except Exception as ex:
        exp_logger.exception("Exception occured: %s", str(ex))
        exp_logger.error(pformat(locals()))
        raise ex
    finally:
        if params.cleanup_fn is not None:
            params.cleanup_fn(params)

    return fname


def run_experiments(exp_params_list: List[ExpParams],
                    exp_log_level: int = logging.INFO+1,
                    n_procs: Optional[int] = None,
                    result_dir: Optional[str] = None,
                    extra_output_dir: Optional[str] = None) -> str:
    """Run series of experiments."""
    print(f"run_experiments - {ray.is_initialized()}")

    exp_start_time = time.time()
    logging.basicConfig(level=exp_log_level, format='%(message)s')

    num_exps = len(exp_params_list)
    logging.log(exp_log_level, "Running %d experiments", num_exps)

    if result_dir is None:
        result_dir = stats_lib.make_dir(exp_params_list[0].env_name)

    logging.log(exp_log_level, "Saving results to dir=%s", result_dir)

    if n_procs is None:
        n_procs = os.cpu_count()
    logging.log(exp_log_level, "Running %d processes", n_procs)

    mp_lock = mp.Lock()

    if n_procs == 1:
        _init_lock(mp_lock)
        for params in exp_params_list:
            run_single_experiment((params, result_dir))
    else:
        args_list = [(params, result_dir) for params in exp_params_list]
        with mp.Pool(
                n_procs, initializer=_init_lock, initargs=(mp_lock,)
        ) as p:
            p.map(run_single_experiment, args_list, 1)

    logging.log(exp_log_level, "Compiling results")
    stats_lib.compile_results(result_dir, extra_output_dir)

    logging.log(
        exp_log_level,
        "Experiment Run time %.2f seconds",
        time.time() - exp_start_time
    )

    return result_dir


def write_experiment_arguments(args: Dict[str, Any], result_dir: str) -> str:
    """Write experiment arguments to file."""
    arg_file = os.path.join(result_dir, EXP_ARG_FILE_NAME)
    with open(arg_file, "w", encoding="utf-8") as fout:
        json.dump(args, fout)
    return arg_file

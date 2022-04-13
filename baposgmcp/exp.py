"""Functions and data structures for running experiments."""
import os
import copy
import time
import random
import logging
from pprint import pformat
import multiprocessing as mp
from typing import (
    List, Optional, Dict, Any, NamedTuple, Callable, Sequence, Set, Tuple
)

import numpy as np

import posggym

from baposgmcp import runner
import baposgmcp.stats as stats_lib
import baposgmcp.render as render_lib
import baposgmcp.policy as policy_lib


LINE_BREAK = "-"*60


class PolicyParams(NamedTuple):
    """Params for a policy in a single experiment run."""
    name: str
    gamma: float
    kwargs: Dict[str, Any]
    # This should take as arguments:
    # [posggym.POSGModel, M.AgentID, float, **kwargs]
    # i.e. [model, agent_id, gamma, kwargs]
    # and return a policy
    # The experiment logger will be added to the kwargs
    init: Callable[..., policy_lib.BasePolicy]


class ExpParams(NamedTuple):
    """Params for a single experiment run."""
    exp_id: int
    env_name: str
    policy_params_list: List[PolicyParams]
    run_config: runner.RunConfig
    tracker_fn: Optional[Callable[[], Sequence[stats_lib.Tracker]]] = None
    render_fn: Optional[Callable[[], Sequence[render_lib.Renderer]]] = None
    stream_log_level: int = logging.INFO
    file_log_level: int = logging.DEBUG


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

    for i in range(num_agents):
        for header in policy_headers:
            if header not in stats[i]:
                stats[i][header] = 'None'

    return stats


def run_single_experiment(args: Tuple[ExpParams, str]) -> str:
    """Run a single experiment and write results to a file."""
    params, result_dir = args
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
        # kwargs = copy.deepcopy(pi_params.kwargs)
        kwargs["logger"] = exp_logger

        pi = pi_params.init(
            env.model, i, pi_params.gamma, **pi_params.kwargs
        )
        policies.append(pi)

    if params.tracker_fn:
        trackers = params.tracker_fn()
    else:
        trackers = stats_lib.get_default_trackers(policies)

    renderers = params.render_fn() if params.render_fn else []

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

    return fname


def run_experiments(exp_params_list: List[ExpParams],
                    exp_log_level: int = logging.INFO+1,
                    n_procs: Optional[int] = None,
                    result_dir: Optional[str] = None,
                    extra_output_dir: Optional[str] = None) -> str:
    """Run series of experiments."""
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

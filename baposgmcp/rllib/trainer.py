import os
import tempfile
from datetime import datetime
from typing import Optional, Callable

import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.logger import NoopLogger, UnifiedLogger


def noop_logger_creator(config):
    """Create a NoopLogger for an rllib trainer."""
    return NoopLogger(config, "")


def custom_log_creator(custom_path: str,
                       custom_str: str,
                       within_default_base_dir: bool = True) -> Callable:
    """Get custom log creator that can be passed to a Trainer.

    In particular `custom_path` specifies the path where results will be
    written by the trainer. If `within_default_base_dir=True` then this will
    be within the RLLIB Default results dir `~/ray_results`.

    `custom_str` is used as the prefix for the logdir.
    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    if within_default_base_dir:
        custom_path = os.path.join(DEFAULT_RESULTS_DIR, custom_path)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        # use mkdtemp to handle race conditions if two processes try create
        # same directory
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


class BAPOSGMCPPPOTrainer(PPOTrainer):
    """Custom Rllib trainer class for the Rllib PPOPolicy.

    Adds functions needed by BAPOSGMCP for experiments, etc.
    """

    def sync_weights(self):
        """Sync weights between all workers.

        This is only implemented so that it's easier to sync weights when
        running with Trainers as ray remote Actors (i.e. when training in
        parallel).
        """
        self.workers.sync_weights()


def get_remote_trainer(env_name: str,
                       trainer_class,
                       policies,
                       policy_mapping_fn,
                       policies_to_train,
                       num_workers: int,
                       num_gpus_per_trainer: float,
                       default_trainer_config,
                       logger_creator: Optional[Callable] = None):
    """Get remote trainer."""
    trainer_remote = ray.remote(
        num_cpus=num_workers,
        num_gpus=num_gpus_per_trainer,
        memory=None,
        object_store_memory=None,
        resources=None
    )(trainer_class)

    trainer_config = dict(default_trainer_config)
    trainer_config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": policies_to_train,
    }

    if num_gpus_per_trainer == 0.0:
        # needed to avoid error
        trainer_config["num_gpus"] = 0.0
    else:
        trainer_config["num_gpus"] = 1.0

    trainer = trainer_remote.remote(
        env=env_name,
        config=trainer_config,
        logger_creator=logger_creator
    )

    return trainer


def get_trainer(env_name: str,
                trainer_class,
                policies,
                policy_mapping_fn,
                policies_to_train,
                default_trainer_config,
                logger_creator: Optional[Callable] = None):
    """Get trainer."""
    trainer_config = dict(default_trainer_config)
    trainer_config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": policies_to_train,
    }

    trainer = trainer_class(
        env=env_name,
        config=trainer_config,
        logger_creator=logger_creator
    )

    return trainer

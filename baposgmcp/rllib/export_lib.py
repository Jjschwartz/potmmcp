import os
import copy
import pickle
import tempfile
from datetime import datetime
from typing import Dict, Sequence, Union

import ray

from baposgmcp import pbt
from baposgmcp.parts import AgentID, PolicyID, Policy
from baposgmcp.rllib.utils import RllibTrainerMap

from baposgmcp.rllib.import_export_utils import (
    TRAINER_CONFIG_FILE, nested_remove
)


def _export_trainer_config(export_dir: str, config: Dict):
    config_path = os.path.join(export_dir, TRAINER_CONFIG_FILE)
    with open(config_path, "wb") as fout:
        pickle.dump(config, fout)


def get_trainer_export_fn(trainer_map: RllibTrainerMap,
                          trainers_remote: bool,
                          config_to_remove: Sequence[Union[str, Sequence[str]]]
                          ) -> pbt.PolicyExportFn:
    """Get function for exporting trained policies to local directory."""

    def export_fn(agent_id: AgentID,
                  policy_id: PolicyID,
                  policy: Policy,
                  export_dir: str):
        if policy_id not in trainer_map[agent_id]:
            # untrained policy, e.g. a random policy
            return

        trainer = trainer_map[agent_id][policy_id]

        if trainers_remote:
            trainer.set_weights.remote(policy)
            ray.get(trainer.save.remote(export_dir))    # type: ignore
            config = ray.get(trainer.get_config.remote())   # type: ignore
        else:
            trainer.set_weights(policy)
            trainer.save(export_dir)
            config = trainer.config

        config = copy.deepcopy(config)

        # this allows removal of unpickalable objects in config
        nested_remove(config, config_to_remove)

        _export_trainer_config(export_dir, config)

    return export_fn


def export_trainers_to_file(parent_dir: str,
                            igraph: pbt.InteractionGraph,
                            trainers: RllibTrainerMap,
                            trainers_remote: bool,
                            save_dir_name: str = "") -> str:
    """Export Rllib trainer objects to file.

    Handles creation of directory to store
    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir_name = f"{save_dir_name}_{timestr}"
    export_dir = tempfile.mkdtemp(prefix=export_dir_name, dir=parent_dir)

    igraph.export_graph(
        export_dir,
        get_trainer_export_fn(
            trainers,
            trainers_remote,
            # remove unpickalable config values
            config_to_remove=[
                "evaluation_config", ["multiagent", "policy_mapping_fn"]
            ]
        )
    )
    return export_dir

import os.path as osp
from typing import Optional

import ray
from ray.tune.logger import pretty_print

from posggym_agents import pbt

from posggym_agents.rllib.utils import RllibTrainerMap
from posggym_agents.rllib.export_lib import export_trainers_to_file
from posggym_agents.rllib.import_lib import (
    TrainerImportArgs, import_igraph_trainers
)


def _check_train_policy_weights(trainer):
    """Check trainer's local and remote worker policies are the same.

    Note will only work for a local trainer (i.e. is in the same process as
    the one calling this function).
    """
    local_weights = trainer.workers.local_worker().get_weights()
    remote_weights = []
    for remote_worker in trainer.workers.remote_workers():
        weights = ray.get(remote_worker.get_weights.remote())
        remote_weights.append(weights)

    for weights in remote_weights:
        for pi_name in local_weights:
            assert pi_name in weights, (
                f"Missing pi {pi_name} from remote weights"
            )
        for pi_name, pi_weights in weights.items():
            assert pi_name in local_weights, (
                f"Additional pi {pi_name} in remote weights"
            )
            if len(pi_weights) == 0:
                continue

            remote_pi_layers = list(pi_weights.keys())
            remote_pi_layers.sort()
            local_pi_layers = list(local_weights[pi_name].keys())
            local_pi_layers.sort()
            assert local_pi_layers == remote_pi_layers, (
                f"Mismatched network layers for {pi_name}: "
                f"{local_pi_layers=} {remote_pi_layers=}"
            )

            for layer in remote_pi_layers:
                remote_pi_layer_weights = pi_weights[layer]
                local_pi_layer_weights = local_weights[pi_name][layer]
                weights_equal = (
                    remote_pi_layer_weights == local_pi_layer_weights
                )
                assert weights_equal.all(), (
                    f"Network weights mismatch for {pi_name=} {layer=}: "
                    f"{local_pi_layer_weights=}\n{remote_pi_layer_weights=}"
                )


def sync_policies(trainers: RllibTrainerMap, igraph: pbt.InteractionGraph):
    """Sync the policies for between trainers based on interactiong graph."""
    agent_ids = list(trainers)
    agent_ids.sort()

    for i, policy_map in trainers.items():
        for policy_k_id, trainer_k in policy_map.items():
            if isinstance(trainer_k, ray.actor.ActorHandle):
                weights = trainer_k.get_weights.remote([policy_k_id])
            else:
                weights = trainer_k.get_weights([policy_k_id])
            igraph.update_policy(i, policy_k_id, weights)

    # swap weights of other agent policies
    for i, policy_map in trainers.items():
        for policy_k_id, trainer_k in policy_map.items():
            for j in agent_ids:
                other_agent_policies = igraph.get_all_policies(
                    i, policy_k_id, j
                )
                # Notes weights here is a dict from policy id to weights
                # ref: https://docs.ray.io/en/master/_modules/ray/rllib/
                #      agents/trainer.html#Trainer.get_weights
                for (policy_j_id, weights) in other_agent_policies:
                    if isinstance(trainer_k, ray.actor.ActorHandle):
                        trainer_k.set_weights.remote(weights)
                        trainer_k.sync_weights.remote()
                    else:
                        trainer_k.set_weights(weights)
                        trainer_k.sync_weights()


def run_training(trainers: RllibTrainerMap,
                 igraph: pbt.InteractionGraph,
                 num_iterations: int,
                 verbose: bool = True):
    """Train Rllib training iterations.

    trainers is Dict[AgentID, Dict[PolicyID, Trainer]]
    """
    agent_ids = list(trainers)
    agent_ids.sort()

    for iteration in range(num_iterations):
        if verbose:
            print(f"== Iteration {iteration} ==")

        sync_policies(trainers, igraph)

        result_futures = {i: {} for i in agent_ids}    # type: ignore
        for i, policy_map in trainers.items():
            for policy_k_id, trainer_k in policy_map.items():
                if isinstance(trainer_k, ray.actor.ActorHandle):
                    result_futures[i][policy_k_id] = trainer_k.train.remote()
                else:
                    result_futures[i][policy_k_id] = trainer_k.train()

        results = {i: {} for i in agent_ids}          # type: ignore
        for i, policy_map in result_futures.items():
            for policy_k_id, future in result_futures[i].items():
                if isinstance(future, ray.ObjectRef):
                    result = ray.get(future)
                else:
                    result = future
                results[i][policy_k_id] = result

        for i, policy_map in results.items():
            for policy_k_id, result_k in policy_map.items():
                if verbose:
                    print(f"-- Agent ID {i}, Policy {policy_k_id} --")
                    print(pretty_print(result_k))


def run_evaluation(trainers: RllibTrainerMap, verbose: bool = True):
    """Run evaluation for policy trainers."""
    if verbose:
        print("== Running Evaluation ==")
    results = {i: {} for i in trainers}
    for i, policy_map in trainers.items():
        results[i] = {}
        for policy_k_id, trainer in policy_map.items():
            if verbose:
                print(f"-- Running Agent ID {i}, Policy {policy_k_id} --")
            results[i][policy_k_id] = trainer.evaluate()

    if verbose:
        print("== Evaluation results ==")
        for i, policy_map in results.items():
            for policy_k_id, result in policy_map.items():
                print(f"-- Agent ID {i}, Policy {policy_k_id} --")
                print(pretty_print(result))

    return results


def continue_training(policy_dir,
                      is_symmetric,
                      trainer_class,
                      trainers_remote: bool,
                      num_iterations: int,
                      seed: Optional[int],
                      num_workers: int,
                      num_gpus: float,
                      save_policies: bool = True,
                      verbose: bool = True):
    """Continue training of saved policies.

    Assumes:
    1. ray has been initialized
    2. training environment has been registered with ray
    """
    trainer_args = TrainerImportArgs(
        trainer_class=trainer_class,
        trainer_remote=trainers_remote,
        num_workers=num_workers,
    )

    igraph, trainers = import_igraph_trainers(
        igraph_dir=policy_dir,
        env_is_symmetric=is_symmetric,
        trainer_args=trainer_args,
        policy_mapping_fn=None,
        extra_config={},
        seed=seed,
        num_gpus=num_gpus
    )
    igraph.display()

    run_training(trainers, igraph, num_iterations, verbose=verbose)

    if save_policies:
        print("== Exporting Graph ==")
        # use same save dir name but with new checkpoint number
        policy_dir_name = osp.basename(osp.normpath(policy_dir))
        name_tokens = policy_dir_name.split("_")[-1]
        if "checkpoint" in name_tokens[-1]:
            try:
                checkpoint_num = int(name_tokens[-1].replace("checkpoint", ""))
                checkpoint = f"checkpoint{checkpoint_num+1}"
            except ValueError:
                checkpoint = name_tokens[-1] + "1"
            name_tokens = name_tokens[:-1]
        else:
            checkpoint = "checkpoint1"
        name_tokens.append(checkpoint)
        save_dir = "_".join(name_tokens)

        export_dir = export_trainers_to_file(
            osp.dirname(osp.normpath(policy_dir)),
            igraph,
            trainers,
            trainers_remote=trainers_remote,
            save_dir_name=save_dir
        )
        print(f"{export_dir=}")

from typing import Optional, Callable

import ray
from ray.tune.logger import pretty_print

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.utils.annotations import override

from baposgmcp import pbt

from baposgmcp.rllib.utils import RllibTrainerMap


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


def _sync_policies(trainers: RllibTrainerMap, igraph: pbt.InteractionGraph):
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

        _sync_policies(trainers, igraph)

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

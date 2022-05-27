from typing import Optional, Callable

import ray
from ray.tune.logger import pretty_print

from baposgmcp import pbt

from baposgmcp.rllib.utils import RllibTrainerMap


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


def run_training(trainers: RllibTrainerMap,
                 igraph: pbt.InteractionGraph,
                 num_iterations: int,
                 verbose: bool = True):
    """Train Rllib training iterations."""
    agent_ids = list(trainers)
    agent_ids.sort()

    for iteration in range(num_iterations):
        if verbose:
            print(f"== Iteration {iteration} ==")

        result_futures = {i: {} for i in agent_ids}    # type: ignore
        for i, policy_map in trainers.items():
            for policy_k_id, trainer_k in policy_map.items():
                result_futures[i][policy_k_id] = trainer_k.train.remote()

        results = {i: {} for i in agent_ids}          # type: ignore
        for i, policy_map in result_futures.items():
            results[i] = {
                policy_k_id: ray.get(future)
                for policy_k_id, future in result_futures[i].items()
            }

        for i, policy_map in results.items():
            for policy_id, result in policy_map.items():
                if verbose:
                    print(f"-- Agent ID {i}, Policy {policy_id} --")
                    print(pretty_print(result))

                igraph.update_policy(
                    i,
                    policy_id,
                    trainers[i][policy_id].get_weights.remote(policy_id)
                )

        # # swap weights of other agent policies
        for i, agent_trainer_map in trainers.items():
            for policy_id, trainer in agent_trainer_map.items():
                for j in agent_ids:
                    if i == j:
                        continue
                    other_agent_policies = igraph.get_all_policies(
                        i, policy_id, j
                    )
                    # Notes weights here is a dict from policy id to weights
                    # ref: https://docs.ray.io/en/master/_modules/ray/rllib/
                    #      agents/trainer.html#Trainer.get_weights
                    for (_, weights) in other_agent_policies:
                        trainer.set_weights.remote(weights)


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

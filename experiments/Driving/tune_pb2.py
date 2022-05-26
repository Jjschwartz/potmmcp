"""Script for tuning rl policies.

Ref:
- https://docs.ray.io/en/master/tune/examples/includes/pb2_ppo_example.html
"""
import random
import argparse

from ray import tune
from ray.tune import sample_from
from ray.tune.registry import register_env
from ray.tune.schedulers.pb2 import PB2

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy

from baposgmcp import pbt
import baposgmcp.rllib as ba_rllib

from exp_utils import registered_env_creator


# NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
NUM_GPUS = 1

# Ref: https://docs.ray.io/en/latest/rllib/rllib-training.html#configuration
RL_TRAINER_CONFIG = {
    # == Rollout worker processes ==
    # A single rollout worker
    "num_workers": 1,
    "num_envs_per_worker": 1,
    # == Trainer process and PPO Config ==
    # ref: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
    "gamma": 0.99,
    "use_critic": True,
    "use_gae": True,
    "lambda": 1.0,
    "kl_coeff": 0.2,
    "rollout_fragment_length": 100,
    "train_batch_size": 2048,
    "sgd_minibatch_size": 256,
    "shuffle_sequences": True,
    "num_sgd_iter": 10,
    "lr": 0.0003,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "model": {
        # === Model Config ===
        # ref: ray-project/ray/blob/releases/1.12.0/rllib/models/catalog.py
        # === Built-in options ===
        # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
        # These are used if no custom model is specified and the input space is
        # 1D. Number of hidden layers to be used.
        "fcnet_hiddens": [256, 256],
        # Activation function descriptor.
        # Supported values are: "tanh", "relu", "swish" (or "silu"),
        # "linear" (or None).
        "fcnet_activation": "tanh",

        # Whether layers should be shared for the value function.
        "vf_share_layers": False,

        # == LSTM ==
        # Whether to wrap the model with an LSTM.
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 20,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
    },
    "entropy_coeff": 0.0,
    "entropy_coeff_schedule": None,
    "clip_param": 0.3,
    "vf_clip_param": 15.0,
    "grad_clip": None,
    "kl_target": 0.01,
    # "trancate_episodes" or "complete_episodes"
    "batch_mode": "truncate_episodes",
    "optimizer": {},
    # == Environment settings ==
    # ...

    # == Deep LEarning Framework Settings ==
    "framework": "torch",
    # == Exploration Settings ==
    "explore": True,
    "exploration_config": {
        "type": "StochasticSampling"
    },
    # == Advanced Rollout Settings ==
    "observation_filter": "NoFilter",
    "metrics_num_episodes_for_smoothing": 100,
    # == Resource Settungs ==
    "num_gpus": 0.0,
}


def _get_env(args):
    return registered_env_creator({"env_name": args.env_name})


def _get_config(args):
    config = dict(RL_TRAINER_CONFIG)
    config["env"] = args.env_name
    config["env_config"] = {"env_name": args.env_name}
    config["log_level"] = args.log_level
    config["seed"] = args.seed
    config["num_cpus_per_worker"] = args.num_cpus_per_worker
    config["num_envs_per_worker"] = args.num_envs_per_worker

    sample_env = _get_env(args)
    # obs and action spaces are the same for both agent in TwoPaths env
    obs_space = sample_env.observation_space["0"]
    act_space = sample_env.action_space["0"]

    policy_random_spec = PolicySpec(RandomPolicy, obs_space, act_space, {})
    policy_ppo_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})

    policy_random_id = pbt.get_klr_policy_id(None, -1, True)
    policy_ppo_id = pbt.get_klr_policy_id(None, 0, True)

    policies = {
        policy_random_id: policy_random_spec,
        policy_ppo_id: policy_ppo_spec
    }

    multiagent_config = {
        "policies": policies,
        "policy_mapping_fn": ba_rllib.default_symmetric_policy_mapping_fn,
        "policies_to_train": [policy_ppo_id]
    }
    config["multiagent"] = multiagent_config

    return config


def main(args):
    """Run stuff."""
    # check env name is valid
    _get_env(args)

    register_env(args.env_name, registered_env_creator)

    config = _get_config(args)

    pb2 = PB2(
        time_attr=args.criteria,
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=args.t_ready,
        quantile_fraction=args.perturb,
        # hyperparameter search space
        hyperparam_bounds={
            "num_sgd_iter": [1, 10],
            "lambda": [0.9, 1.0],
            "clip_param": [0.1, 0.5],
            "lr": [1e-3, 1e-5],
            "train_batch_size": [1000, 60000],
        },
    )

    config.update({
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 128,
        "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
        "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
        "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
        "train_batch_size": sample_from(
            lambda spec: random.randint(1000, 40000)
        ),
    })

    tune.run(
        "PPO",
        name=f"PPO_{args.env_name}",
        scheduler=pb2,
        verbose=1,
        num_samples=args.num_samples,
        stop={args.criteria: args.tune_max},
        local_dir=f"~/ray_results/{args.env_name}",
        config=config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env_name", type=str,
        help="Name of the environment to train on."
    )
    parser.add_argument(
        "--log_level", type=str, default='WARN',
        help="Log level"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of workers"
    )
    parser.add_argument(
        "--num_gpus", type=float, default=1.0,
        help="Number of GPUs to use."
    )
    parser.add_argument(
        "--num_cpus_per_worker", type=int, default=1,
        help="Number of CPUs per worker."
    )
    parser.add_argument(
        "--num_envs_per_worker", type=int, default=1,
        help="Number of envs per worker."
    )
    parser.add_argument(
        "--num_samples", type=int, default=4,
        help="Number of tune samples to run simoultaneously (?)."
    )
    parser.add_argument(
        "--criteria", type=str, default="timesteps_total",
        help=(
            "Tune time criteria ('training_iteration', 'time_total_s',"
            "timesteps_total)"
        )
    )
    parser.add_argument(
        "--tune_max", type=int, default=500000,
        help="Total tune period, for chosen criteria."
    )
    parser.add_argument(
        "--perturb", type=float, default=0.25,
        help="PB2 quantile fraction."
    )
    parser.add_argument(
        "--t_ready", type=int, default=50000,
        help="Perturbation interval."
    )
    main(parser.parse_args())

import pathlib
import os.path as osp

import posggym
from posggym.wrappers import FlattenObservation
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv


EXP_BASE_DIR = osp.dirname(osp.abspath(__file__))
EXP_RESULTS_DIR = osp.join(EXP_BASE_DIR, "results")
EXP_SAVED_RESULTS_DIR = osp.join(EXP_BASE_DIR, "saved_results")
EXP_RL_POLICY_DIR = osp.join(EXP_BASE_DIR, "rl_policies")

pathlib.Path(EXP_RESULTS_DIR).mkdir(exist_ok=True)
pathlib.Path(EXP_RL_POLICY_DIR).mkdir(exist_ok=True)


def registered_env_creator(config):
    """Create a new posggym registered Driving Environment."""
    env = posggym.make(config["env_name"], **{"seed": config["seed"]})
    env = FlattenObservation(env)
    return RllibMultiAgentEnv(env)


def get_rllib_env(args) -> RllibMultiAgentEnv:
    """Create RllibMultiAgentEnv from commandline args."""
    return registered_env_creator(
        {"env_name": args.env_name, "seed": args.seed}
    )


def get_base_env(args) -> posggym.Env:
    """Create POSGGYM base (unwrapped)  environment from commandline args."""
    return posggym.make(args.env_name, **{"seed": args.seed})


# Ref: https://docs.ray.io/en/latest/rllib/rllib-training.html#configuration
RL_TRAINER_CONFIG = {
    # == Resource Settungs ==
    # Number of GPUs per trainer
    #   num_gpus controls the GPU allocation for the learner process
    #   we want the learner using the GPU as it does batch inference
    #   For general training we set this to 1.0 and control resource allocation
    #   via ray.remote.
    #   For policy evaluation this should typically be set to 0.0 since using
    #   the GPU is only faster when doing batched inference.
    "num_gpus": 1.0,
    # Number of cpus assigned to each worker
    "num_cpus_per_worker": 1.0,
    # Number of gpus per rollout worker.
    # this should be 0 since we want rollout workers using CPU since they
    # don't do batch inference
    "num_gpus_per_worker": 0.0,
    # == Rollout worker processes ==
    # Number of rollout workers per trainer
    #   a value of 1 means each trainer will have two processes:
    #   1. a learner process which updates network
    #   2. a rollout worker which collects trajectories
    "num_workers": 1,
    # Number of environments to run simoultaneously per rollout worker
    "num_envs_per_worker": 4,

    # == Trainer process and PPO Config ==
    # ref: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
    "gamma": 0.99,
    "use_critic": True,
    "use_gae": True,
    "lambda": 0.9,
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    # this is automatically adjusted to = train_batch_size /  num_workers
    "rollout_fragment_length": 200,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 2048,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 256,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 10,
    # Stepsize of SGD.
    "lr": 0.0003,
    # Learning rate schedule.
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "model": {
        # === Model Config ===
        # ref: ray-project/ray/blob/releases/1.12.0/rllib/models/catalog.py
        # === Built-in options ===
        # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
        # These are used if no custom model is specified and the input space is
        # 1D. Number of hidden layers to be used.
        "fcnet_hiddens": [64, 32],
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
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": None,
    "clip_param": 0.3,
    "vf_clip_param": 30.0,
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
}

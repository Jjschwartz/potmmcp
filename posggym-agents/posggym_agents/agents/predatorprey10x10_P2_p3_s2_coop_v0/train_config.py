import copy


def get_rl_training_config(env_id: str, seed: int, log_level: str):
    """Get the Rllib agent config for an agent being trained."""
    config = copy.deepcopy(RL_TRAINER_CONFIG)
    config["log_level"] = log_level
    config["seed"] = seed
    config["env_config"] = {
        "env_name": env_id,
        "seed": seed,
        "flatten_obs": True
    }
    config["explore"] = True
    config["exploration_config"] = {
        "type": "StochasticSampling",
        # add some random timesteps to get agents away from initial "safe"
        # starting positions
        "random_timesteps": 5
    }
    return config


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
    "gamma": 0.999,
    "use_critic": True,
    "use_gae": True,
    "lambda": 0.95,
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    # this is automatically adjusted to = train_batch_size /  num_workers
    "rollout_fragment_length": 128,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 2048,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 512,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 2,
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
    "entropy_coeff": 0.001,
    "entropy_coeff_schedule": None,
    "clip_param": 0.2,
    # max return is 1.0 so clip as such
    "vf_clip_param": 1.0,
    "grad_clip": 5.0,
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
    # == Logging ==
    # default is "WARN". Options (in order of verbosity, most to least) are:
    # "DEBUG", "INFO", "WARN", "ERROR"
    "log_level": "WARN"
}

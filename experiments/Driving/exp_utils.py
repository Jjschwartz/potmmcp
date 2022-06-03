import pathlib
import tempfile
import os.path as osp
from datetime import datetime
from typing import Optional, List, Dict, Callable

from ray.tune.logger import NoopLogger
from ray.rllib.agents.ppo import PPOTrainer

import posggym
import posggym.model as M
from posggym.wrappers import FlattenObservation
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv

from baposgmcp import pbt
import baposgmcp.exp as exp_lib
import baposgmcp.rllib as ba_rllib
import baposgmcp.policy as ba_policy_lib
from baposgmcp.config import BASE_RESULTS_DIR


EXP_BASE_DIR = osp.dirname(osp.abspath(__file__))
EXP_BASE_SAVE_DIR = osp.join(BASE_RESULTS_DIR, "Driving")
EXP_RESULTS_DIR = osp.join(EXP_BASE_SAVE_DIR, "results")
EXP_RL_POLICY_DIR = osp.join(EXP_BASE_SAVE_DIR, "rl_policies")

pathlib.Path(EXP_BASE_SAVE_DIR).mkdir(exist_ok=True)
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


def get_base_env(env_name: str, seed: Optional[int]) -> posggym.Env:
    """Create POSGGYM base (unwrapped)  environment from commandline args."""
    return posggym.make(env_name, **{"seed": seed})


def get_training_logger_creator(parent_dir: str,
                                env_name: str,
                                seed: Optional[int],
                                suffix: Optional[str]) -> Callable:
    """Get logger creator for training.

    `parent_dir` is the directory within which files will be logged.
    This will be a subdirectory of `~/ray_results/Driving`
    """
    custom_path = osp.join("Driving", parent_dir)

    custom_str = f"PPOTrainer_{env_name}"
    if seed is not None:
        custom_str += f"_seed={seed}"
    if suffix is not None:
        custom_str += f"_{suffix}"

    return ba_rllib.custom_log_creator(custom_path, custom_str, True)


def get_result_dir(prefix, root_dir: Optional[str] = None) -> str:
    """Get experiment result dir. Handles creating dir."""
    if root_dir is None:
        root_dir = EXP_RESULTS_DIR
    else:
        assert osp.isdir(root_dir)

    time_str = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir_name = f"{prefix}_{time_str}"
    result_dir = tempfile.mkdtemp(prefix=result_dir_name, dir=root_dir)
    return result_dir


def _trainer_make_fn(config):
    return PPOTrainer(
        env=config["env_config"]["env_name"],
        config=config,
        logger_creator=lambda c: NoopLogger(c, "")
    )


def import_rllib_policy(policy_dir,
                        policy_id,
                        agent_id,
                        num_gpus: float = 0.0,
                        num_workers: int = 0,
                        env_name: Optional[str] = None,
                        log_level: str = "ERROR"):
    """Import rllib policy from file.

    Recommended to use:
    - num_gpus=0.0 if only using policy for inference on single observation
      (i.e. for rollouts not batched training).
    - num_workers=0.0 agains if only using policy for inference on single
      observation

    Include env_name in case you are running on an environment that is
    different to the one used for training and the original training env has
    not been registered with rllib.

    """
    extra_config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "log_level": log_level,
        "num_envs_per_worker": 1,
        # disables logging of CPU and GPU usage
        "log_sys_usage": False,
    }

    if env_name:
        extra_config["env_config"] = {
            "env_name": env_name
        }

    return ba_rllib.import_policy(
        policy_id=policy_id,
        igraph_dir=policy_dir,
        env_is_symmetric=True,
        agent_id=agent_id,
        trainer_make_fn=_trainer_make_fn,
        policy_mapping_fn=None,
        trainers_remote=False,
        extra_config=extra_config
    )


def load_agent_policy(policy_dir: str,
                      policy_id: str,
                      agent_id: M.AgentID,
                      env_name: str,
                      gamma: float,
                      seed: Optional[int] = None
                      ) -> ba_rllib.PPORllibPolicy:
    """Load BAPOSGMCP rllib agent policy from file.

    Note this function differs from the import_rllib_policy function. This
    function calls the import_rllib_policy function and then handles wrapping
    the rllib policy within a BAPOSGMCP policy, so it's compatible with the
    BAPOSGMCP code.
    """
    rllib_policy = import_rllib_policy(
        policy_dir,
        policy_id,
        agent_id,
        num_gpus=0.0,
        num_workers=0,
        env_name=env_name
    )

    sample_env = get_base_env(env_name, seed)
    env_model = sample_env.unwrapped.model
    obs_space = env_model.obs_spaces[agent_id]
    preprocessor = ba_rllib.get_flatten_preprocessor(obs_space)

    policy = ba_rllib.PPORllibPolicy(
        model=env_model,
        ego_agent=agent_id,
        gamma=gamma,
        policy=rllib_policy,
        policy_id=policy_id,
        preprocessor=preprocessor,
    )
    return policy


def rllib_policy_init_fn(model, ego_agent, gamma, **kwargs):
    """Initialize a PPORllibPolicy from kwargs.

    Expects kwargs:
      'policy_dir' - save directory of rllib policy
      'policy_id' - ID of the rllib policy

    """
    policy_dir = kwargs.pop("policy_dir")

    env_name = None
    if "env_name" in kwargs:
        env_name = kwargs.pop("env_name")

    preprocessor = ba_rllib.get_flatten_preprocessor(
        model.obs_spaces[ego_agent]
    )

    pi = import_rllib_policy(
        policy_dir,
        kwargs["policy_id"],
        ego_agent,
        num_gpus=0.0,
        num_workers=0,
        env_name=env_name
    )

    return ba_rllib.PPORllibPolicy(
        model=model,
        ego_agent=ego_agent,
        gamma=gamma,
        policy=pi,
        preprocessor=preprocessor,
        **kwargs
    )


def load_agent_policy_params(policy_dir: str,
                             gamma: float,
                             env_name: Optional[str] = None,
                             include_random_policy: bool = True
                             ) -> List[exp_lib.PolicyParams]:
    """Load agent rllib policy params from file.

    Note, this function imports policy params such that policies will only be
    loaded from file only when the policy is to be used in an experiment. This
    saves on memory usage and also ensures a different policy object is used
    for each experiment run.
    """
    igraph = ba_rllib.import_igraph(policy_dir, True)

    info = {
        # this helps differentiate policies trained on different
        # envs or from different training runs/seeds
        "policy_dir": policy_dir
    }

    policy_params_list = []
    random_policy_added = False
    for policy_id in igraph.policies[pbt.InteractionGraph.SYMMETRIC_ID]:
        if "-1" in policy_id:
            policy_params = exp_lib.PolicyParams(
                name="RandomPolicy",
                gamma=gamma,
                kwargs={"policy_id": policy_id},
                init=ba_policy_lib.RandomPolicy,
                info=info
            )
            random_policy_added = True
        else:
            policy_params = exp_lib.PolicyParams(
                name=f"PPOPolicy_{policy_id}",
                gamma=gamma,
                kwargs={
                    "policy_dir": policy_dir,
                    "policy_id": policy_id,
                    "env_name": env_name
                },
                init=rllib_policy_init_fn,
                info=info
            )
        policy_params_list.append(policy_params)

    if include_random_policy and not random_policy_added:
        policy_params = exp_lib.PolicyParams(
            name="RandomPolicy",
            gamma=gamma,
            kwargs={"policy_id": "pi_-1"},
            init=ba_policy_lib.RandomPolicy,
            info=info
        )
        policy_params_list.append(policy_params)

    return policy_params_list


def load_agent_policies(agent_id: int,
                        env_name: str,
                        policy_dir: str,
                        gamma: float,
                        include_random_policy: bool = False,
                        env_seed: Optional[int] = None
                        ) -> Dict[str, ba_policy_lib.BasePolicy]:
    """Load agent rllib policies from file."""
    sample_env = get_base_env(env_name, env_seed)
    env_model = sample_env.unwrapped.model

    _, policy_map = ba_rllib.import_igraph_policies(
        igraph_dir=policy_dir,
        env_is_symmetric=True,
        trainer_make_fn=_trainer_make_fn,
        trainers_remote=False,
        policy_mapping_fn=None,
        extra_config={
            # only using policies for rollouts so no GPU
            "num_gpus": 0.0,
            # no need for seperate rollout workers either
            "num_workers": 0,
            "log_level": "ERROR",
            "env_config": {"env_name": env_name},
            # disables logging of CPU and GPU usage
            "log_sys_usage": False,
        }
    )

    policies_map = {}
    random_policy_added = False
    symmetric_agent_id = pbt.InteractionGraph.SYMMETRIC_ID
    for policy_id, policy in policy_map[symmetric_agent_id].items():
        if "-1" in policy_id:
            new_policy = ba_policy_lib.RandomPolicy(
                env_model, agent_id, gamma
            )
            random_policy_added = True
        else:
            obs_space = env_model.obs_spaces[agent_id]
            preprocessor = ba_rllib.get_flatten_preprocessor(obs_space)
            new_policy = ba_rllib.PPORllibPolicy(
                model=env_model,
                ego_agent=agent_id,
                gamma=gamma,
                policy=policy,
                policy_id=policy_id,
                preprocessor=preprocessor,
            )
        policies_map[policy_id] = new_policy

    if include_random_policy and not random_policy_added:
        new_policy = ba_policy_lib.RandomPolicy(
            env_model,
            agent_id,
            gamma,
            policy_id="pi_-1"
        )
        policies_map["pi_-1"] = new_policy

    return policies_map


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
    # == Logging ==
    # default is "WARN". Options (in order of verbosity, most to least) are:
    # "DEBUG", "INFO", "WARN", "ERROR"
    "log_level": "WARN"
}

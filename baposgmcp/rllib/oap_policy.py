"""Custom Rllib Policies.

These are policies that subclass existing Rllib policies for the purpose of
adding custom functionallity.
"""
from typing import List, Union, Type, Dict

import gym
import numpy as np
import torch
import torch.nn as nn

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import (
    compute_gae_for_sample_batch,
    Postprocessing,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
)


OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"
OAP_PREDS = "oap_preds"


class OAPTorchModel(TorchFC):
    """Multi-agent FCNet that implements a other-agent policy prediction.

    Assumes:
    - the other agent policy is discrete
    - the other agent's action space is that same as the ego agent action space

    Maps ego observations -> other agent policy (dist over other agent actions)

    Note, the handling of recurrency is done automatically by Rllib using the
    auto-lstm wrapper (which also optionally handles passing action and rewards
    with the obs)

    This class basically adds an additional head to the TorchFC class.

    Ref:
    https://github.com/ray-project/ray/blob/releases/1.12.0/rllib/examples/models/centralized_critic_models.py
    https://github.com/ray-project/ray/blob/releases/1.12.0/rllib/models/torch/recurrent_net.py
    https://github.com/ray-project/ray/blob/releases/1.12.0/rllib/models/torch/fcnet.py

    Config:
      "oap_share_layers": bool
         whether to share layers between OAP head and policy head
    """

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: Dict,
                 name: str,
                 **customized_model_kwargs):
        assert isinstance(action_space, gym.spaces.Discrete)
        # assert action_space.n == num_outputs
        # initialize the base model for policy and value function
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        if num_outputs is None:
            num_outputs = action_space.n

        # Use same size network for OAP head
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            # use final layer activation, in case no hidden layers
            activation = model_config.get("post_fcnet_activation")

        if "oap_share_layers" in customized_model_kwargs:
            self.oap_share_layers = customized_model_kwargs["oap_share_layers"]
        else:
            # try get it from model config
            self.oap_share_layers = (
                model_config["custom_model_config"]["oap_share_layers"]
            )

        self._oap_branch_seperate = None
        if not self.oap_share_layers:
            # Build a parellel set of hidden layers for the OAP net
            # OAP maps (obs) -> oap_pred
            prev_oap_layer_size = int(np.product(obs_space.shape))
            oap_layers = []

            # create layers 0 to second last
            for size in hiddens:
                oap_layers.append(
                    SlimFC(
                        in_size=prev_oap_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)
                    )
                )
                prev_oap_layer_size = size
            self._oap_branch_seperate = nn.Sequential(*oap_layers)

        # the final OAP layer without activation
        self._oap_logits = SlimFC(
            in_size=hiddens[-1],
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None
        )

    def oap_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._oap_branch_seperate:
            oap_logits = self._oap_logits(
                self._oap_branch_seperate(self._last_flat_in)
            )
        else:
            oap_logits = self._oap_logits(self._features)
        return oap_logits


class OAPLSTMTorchModel(TorchRNN, nn.Module):
    """Same as OAPTorchModel but recurrent.

    Basically wraps the OAPTorchModel with a recurrent layer.

    Adapted from the Rllib LSTMWrapper class.

    Note when using LSTM the same trunk is shared leading into the
    LSTM module before splitting into the different heads:
    FCNet -> LSTM -> FCnet policy
                  -> FCnet value
                  -> FCnet oap

    Ref:
    https://github.com/ray-project/ray/blob/releases/1.12.0/rllib/models/torch/recurrent_net.py
    https://github.com/ray-project/ray/blob/releases/1.12.0/rllib/models/torch/fcnet.py
    https://github.com/ray-project/ray/blob/releases/1.12.0/rllib/examples/models/rnn_model.py
    """

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: Dict,
                 name: str):
        assert isinstance(action_space, gym.spaces.Discrete)
        nn.Module.__init__(self)
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.cell_size = model_config["lstm_cell_size"]
        self.time_major = model_config.get("_time_major", False)
        self.use_prev_action = model_config["lstm_use_prev_action"]
        self.use_prev_reward = model_config["lstm_use_prev_reward"]

        assert not self.use_prev_action, (
            "lstm_use_prev_action not yet supported for OAPLSTMTorchModel"
        )
        assert not self.use_prev_reward, (
            "lstm_use_prev_reward not yet supported for OAPLSTMTorchModel"
        )

        # Build the Module from fc + LSTM + 3xfc (action + value + oap out)
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            # use final layer activation, in case no hidden layers
            activation = model_config.get("post_fcnet_activation")

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None
        self.num_outputs = num_outputs

        # Create the shared trunk before the LSTM.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        self._hidden_layers = nn.Sequential(*layers)

        # The LSTM
        self.lstm = nn.LSTM(
            prev_layer_size, self.cell_size, batch_first=not self.time_major
        )
        prev_layer_size = self.cell_size

        # The policy head
        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None
        )
        # Value head
        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # OAP head
        self._oap_logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None
        )

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        # Place hidden states on same device as model.
        linear = next(self._logits._model.children())
        h = [
            linear.weight.new(1, self.cell_size).zero_().squeeze(0),
            linear.weight.new(1, self.cell_size).zero_().squeeze(0),
        ]
        return h

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        x = self._hidden_layers(inputs)
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action_out = self._logits(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._features), [-1])

    def oap_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        oap_logits = self._oap_logits(self._features)
        oap_logits = torch.reshape(oap_logits, [-1, self.num_outputs])
        return oap_logits


def register_oap_model() -> str:
    """Register the OAP model with rllib."""
    oap_model_name = "oap_model"
    ModelCatalog.register_custom_model(oap_model_name, OAPTorchModel)
    return oap_model_name


def register_lstm_oap_model() -> str:
    """Register the OAP LSTM model with rllib."""
    oap_model_name = "oap_lstm_model"
    ModelCatalog.register_custom_model(oap_model_name, OAPLSTMTorchModel)
    return oap_model_name


def oap_postprocessing(policy,
                       sample_batch,
                       other_agent_batches,
                       episode=None):
    """Other-agent policy Postprocessing.

    Adds other-agent obs/act to the training batch along with usual PPO batch
    stuff.

    Reference:
    github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py
    """
    assert policy.config["framework"] == "torch"
    if policy.oap_policy_initialized:
        assert other_agent_batches is not None
        [(_, opponent_batch)] = list(other_agent_batches.values())

        # record the opponent actions in the trajectory
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS]
        )

    return sample_batch


class OAPPPOTorchPolicy(PPOTorchPolicy):
    """Other Agent Policy PPO Torch Policy.

    Extends the PPO Torch Policy by adding an additional head which predicts
    the action distribution of the other agent in the environment given the
    history of the ego agent.

    This additional other agent policy head is trained using the joint history
    information for all agents. During evaluation it can be used for
    model-based or not at all (i.e. it's treated as an auxiliary loss during
    training only.)

    Config
      "oap_loss_coeff": float
          Coefficient for the OAP component of the loss
    """

    def __init__(self, observation_space, action_space, config):
        # flag used in postprocessing function so it knows when policy has
        # been initialized of not (which happens in super().__init__
        self.oap_policy_initialized = False
        super().__init__(observation_space, action_space, config)
        self.oap_policy_initialized = True

    def compute_oap(self) -> TensorType:
        """Compute other-agent policy."""
        return self.model.oap_function()

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        with torch.no_grad():
            sample_batch = oap_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

    @override(PPOTorchPolicy)
    def loss(self,
             model: ModelV2,
             dist_class: Type[ActionDistribution],
             train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        """Get OAP+PPO Loss.

        This is essentially copy and pasted from PPOTorchPolicy and then the
        OAP loss is added on.

        Ref:
        https://github.com/ray-project/ray/blob/releases/1.12.0/rllib/agents/ppo/ppo_torch_policy.py
        """
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)
        oap_pred_action_dist = model.oap_function()

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
                logp_ratio,
                1 - self.config["clip_param"],
                1 + self.config["clip_param"]
            ),
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(
                vf_loss, 0, self.config["vf_clip_param"]
            )
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            vf_loss_clipped = mean_vf_loss = 0.0

        # Compute OAP loss
        oap_targets = train_batch[OPPONENT_ACTION]
        oap_loss = nn.functional.cross_entropy(
            oap_pred_action_dist, oap_targets, reduction='none'
        )

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
            + self.config["oap_loss_coeff"] * oap_loss
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], model.value_function()
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        # Add OAP loss
        model.tower_stats["oap_loss"] = oap_loss

        print("loss computed")
        return total_loss

    @override(PPOTorchPolicy)
    def extra_grad_info(self,
                        train_batch: SampleBatch) -> Dict[str, TensorType]:
        grad_info = super().extra_grad_info(train_batch)
        oap_grad_info = convert_to_numpy({
            "oap_loss": torch.mean(
                torch.stack(self.get_tower_stats("oap_loss"))
            )
        })
        grad_info.update(oap_grad_info)
        return grad_info


class OAPPPOTrainer(PPOTrainer):
    """Rllib trainer class for the OAPPPOPolicy class."""

    @override(PPOTrainer)
    def get_default_policy_class(self, config):
        return OAPPPOTorchPolicy

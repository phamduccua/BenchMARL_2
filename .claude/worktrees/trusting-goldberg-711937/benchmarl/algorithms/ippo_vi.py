#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.optim as optim
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import IndependentNormal, ProbabilisticActor, TanhNormal
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig
from benchmarl.algorithms.ippo import IppoConfig
from benchmarl.experiment.callback import Callback


class VIOptimizer(optim.Optimizer):
    """Variational Inference-style optimizer with gradient normalization and
    anchor-based regularization (proximal gradient).

    Each :meth:`step` applies::

        grad_regularized = grad / ||grad|| + tau * (params - anchor)
        params -= lr * grad_regularized

    Call :meth:`set_anchor` once before the sequence of optimizer steps that
    belong to the *same* data batch so that the regularization term pulls the
    updated parameters back toward the beginning-of-batch values.
    """

    def __init__(self, params, lr: float, tau: float = 0.1):
        defaults = dict(lr=lr, tau=tau)
        super().__init__(params, defaults)

    def set_anchor(self):
        """Save a snapshot of the current parameters as the anchor point."""
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["anchor"] = p.data.clone()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                tau = group["tau"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "anchor" not in state:
                        raise RuntimeError(
                            "VIOptimizer: call set_anchor() before the first step()."
                        )
                    x_anchor = state["anchor"]
                    grad_norm = torch.norm(p.grad) + 1e-8
                    normalized_grad = p.grad / grad_norm
                    grad_regularized = normalized_grad + tau * (p.data - x_anchor)
                    p.data.add_(grad_regularized, alpha=-lr)

        return loss


class VIOptimizerCallback(Callback):
    """Callback that replaces the actor (``loss_objective``) optimizer with
    :class:`VIOptimizer` for every agent group and calls
    :meth:`VIOptimizer.set_anchor` before each training round.

    Add this callback to your experiment when using :class:`IppoVI`::

        experiment = Experiment(
            ...
            algorithm_config=IppoVIConfig(...),
            callbacks=[VIOptimizerCallback()],
        )
    """

    def __init__(self):
        super().__init__()
        self._vi_optimizers: Dict[str, VIOptimizer] = {}

    def on_setup(self):
        exp = self.experiment
        for group in exp.group_map.keys():
            if "loss_objective" not in exp.optimizers[group]:
                continue
            old_opt = exp.optimizers[group]["loss_objective"]
            # Collect actor parameter groups from the old Adam optimizer
            vi_opt = VIOptimizer(
                [{"params": pg["params"]} for pg in old_opt.param_groups],
                lr=exp.config.lr,
                tau=exp.algorithm_config.vi_tau,
            )
            exp.optimizers[group]["loss_objective"] = vi_opt
            self._vi_optimizers[group] = vi_opt

    def on_batch_collected(self, batch: TensorDictBase):  # noqa: ARG002
        """Set anchor before training steps begin for this batch."""
        for vi_opt in self._vi_optimizers.values():
            vi_opt.set_anchor()


class IppoVI(Algorithm):
    """Independent PPO (from `https://arxiv.org/abs/2011.09533 <https://arxiv.org/abs/2011.09533>`__).

    Args:
        share_param_critic (bool): Whether to share the parameters of the critics withing agent groups
        clip_epsilon (scalar): weight clipping threshold in the clipped PPO loss equation.
        entropy_coef (scalar): entropy multiplier when computing the total loss.
        critic_coef (scalar): critic loss multiplier when computing the total
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1".
        lmbda (float): The GAE lambda
        scale_mapping (str): positive mapping function to be used with the std.
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        use_tanh_normal (bool): if ``True``, use TanhNormal as the continuyous action distribution with support bound
            to the action domain. Otherwise, an IndependentNormal is used.
        minibatch_advantage (bool): if ``True``, advantage computation is perfomend on minibatches of size
            ``experiment.config.on_policy_minibatch_size`` instead of the full
            ``experiment.config.on_policy_collected_frames_per_batch``, this helps not exploding memory usage

    """

    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        minibatch_advantage: bool,
        vi_tau: float = 0.1,
        **kwargs
    ):
        self.vi_tau = vi_tau
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal
        self.minibatch_advantage = minibatch_advantage

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # Loss
        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        # IPPO uses the same stochastic actor for collection
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(
                -self.experiment.config.train_minibatch_size(self.on_policy)
                // batch.shape[1]
            )
        else:
            increment = batch.batch_size[0] + 1
        last_start_index = 0
        start_index = increment
        minibatches = []
        while last_start_index < batch.shape[0]:
            minimbatch = batch[last_start_index:start_index]
            minibatches.append(minimbatch)
            with torch.no_grad():
                loss.value_estimator(
                    minimbatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            last_start_index = start_index
            start_index += increment

        batch = torch.cat(minibatches, dim=0)
        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss_vals.set(
            "loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        )
        del loss_vals["loss_entropy"]
        return loss_vals

    #####################
    # Custom new methods
    #####################

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])

        critic_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
        critic_output_spec = Composite(
            {
                group: Composite(
                    {"state_value": Unbounded(shape=(n_agents, 1))},
                    shape=(n_agents,),
                )
            }
        )
        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=False,
            input_has_agent_dim=True,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )

        return value_module

import importlib
_has_hydra = importlib.util.find_spec("hydra") is not None

@dataclass
class IppoVIConfig(IppoConfig):
    vi_tau: float = 0.1

    @staticmethod
    def associated_class():
        return IppoVI

# Import và đăng ký có điều kiện — giống cách BenchMARL làm
if _has_hydra:
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(
        group="algorithm",
        name="ippo_vi_config",
        node=IppoVIConfig,
    )

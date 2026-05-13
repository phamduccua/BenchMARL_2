#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""IPPO with VIOptimizer — gradient normalization **disabled**.

The standard :class:`~benchmarl.algorithms.ippo_vi.VIOptimizer` applies::

    grad_reg = grad / ‖grad‖ + τ·(θ − θ_anchor)
    θ ← θ − lr · grad_reg

This variant skips the normalization::

    grad_reg = grad + τ·(θ − θ_anchor)
    θ ← θ − lr · grad_reg

Usage::

    from benchmarl.algorithms.ippo_vi_no_norm import (
        IppoVINoNormConfig,
        VIOptimizerNoNormCallback,
    )

    experiment = Experiment(
        ...
        algorithm_config=IppoVINoNormConfig(vi_tau=0.1, ...),
        callbacks=[VIOptimizerNoNormCallback()],
    )
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.optim as optim
from tensordict import TensorDictBase

from benchmarl.algorithms.ippo_vi import IppoVI, IppoVIConfig
from benchmarl.experiment.callback import Callback


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class VIOptimizerNoNorm(optim.Optimizer):
    """VI-style proximal gradient optimizer **without** gradient normalization.

    Update rule::

        grad_reg = grad + τ·(θ − θ_anchor)
        θ ← θ − lr · grad_reg

    Call :meth:`set_anchor` once per data batch before the optimizer steps.
    """

    def __init__(self, params, lr: float, tau: float = 0.1):
        defaults = dict(lr=lr, tau=tau)
        super().__init__(params, defaults)

    def set_anchor(self):
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
                            "VIOptimizerNoNorm: call set_anchor() before step()."
                        )
                    grad_reg = p.grad + tau * (p.data - state["anchor"])
                    p.data.add_(grad_reg, alpha=-lr)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────

class VIOptimizerNoNormCallback(Callback):
    """Replaces the actor optimizer with :class:`VIOptimizerNoNorm` and calls
    :meth:`set_anchor` before each training round."""

    def __init__(self):
        super().__init__()
        self._vi_optimizers: Dict[str, VIOptimizerNoNorm] = {}

    def on_setup(self):
        exp = self.experiment
        for group in exp.group_map.keys():
            if "loss_objective" not in exp.optimizers[group]:
                continue
            old_opt = exp.optimizers[group]["loss_objective"]
            vi_opt = VIOptimizerNoNorm(
                [{"params": pg["params"]} for pg in old_opt.param_groups],
                lr=exp.config.lr,
                tau=exp.algorithm_config.vi_tau,
            )
            exp.optimizers[group]["loss_objective"] = vi_opt
            self._vi_optimizers[group] = vi_opt

    def on_batch_collected(self, batch: TensorDictBase):  # noqa: ARG002
        for vi_opt in self._vi_optimizers.values():
            vi_opt.set_anchor()


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm & Config
# ─────────────────────────────────────────────────────────────────────────────

class IppoVINoNorm(IppoVI):
    """IPPO with :class:`VIOptimizerNoNorm` (anchor regularization, no grad-norm).

    Optimizer difference is handled by :class:`VIOptimizerNoNormCallback`.
    """


@dataclass
class IppoVINoNormConfig(IppoVIConfig):
    @staticmethod
    def associated_class():
        return IppoVINoNorm

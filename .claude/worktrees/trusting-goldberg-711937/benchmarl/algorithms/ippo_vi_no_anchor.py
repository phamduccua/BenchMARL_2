#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""IPPO with VIOptimizer — anchor regularization **disabled**.

The standard :class:`~benchmarl.algorithms.ippo_vi.VIOptimizer` applies::

    grad_reg = grad / ‖grad‖ + τ·(θ − θ_anchor)
    θ ← θ − lr · grad_reg

This variant drops the anchor term, keeping only gradient normalization::

    grad_reg = grad / ‖grad‖
    θ ← θ − lr · grad_reg

Usage::

    from benchmarl.algorithms.ippo_vi_no_anchor import (
        IppoVINoAnchorConfig,
        VIOptimizerNoAnchorCallback,
    )

    experiment = Experiment(
        ...
        algorithm_config=IppoVINoAnchorConfig(...),
        callbacks=[VIOptimizerNoAnchorCallback()],
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

class VIOptimizerNoAnchor(optim.Optimizer):
    """VI-style normalized gradient optimizer **without** anchor regularization.

    Update rule::

        θ ← θ − lr · (grad / ‖grad‖)

    No anchor state needs to be set; ``set_anchor()`` is a no-op kept for
    API compatibility with the full :class:`~benchmarl.algorithms.ippo_vi.VIOptimizer`.
    """

    def __init__(self, params, lr: float):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def set_anchor(self):
        """No-op — anchor regularization is disabled in this variant."""

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad_norm = torch.norm(p.grad) + 1e-8
                    p.data.add_(p.grad / grad_norm, alpha=-lr)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────

class VIOptimizerNoAnchorCallback(Callback):
    """Replaces the actor optimizer with :class:`VIOptimizerNoAnchor`."""

    def __init__(self):
        super().__init__()
        self._vi_optimizers: Dict[str, VIOptimizerNoAnchor] = {}

    def on_setup(self):
        exp = self.experiment
        for group in exp.group_map.keys():
            if "loss_objective" not in exp.optimizers[group]:
                continue
            old_opt = exp.optimizers[group]["loss_objective"]
            vi_opt = VIOptimizerNoAnchor(
                [{"params": pg["params"]} for pg in old_opt.param_groups],
                lr=exp.config.lr,
            )
            exp.optimizers[group]["loss_objective"] = vi_opt
            self._vi_optimizers[group] = vi_opt

    def on_batch_collected(self, batch: TensorDictBase):  # noqa: ARG002
        # Kept for symmetry; VIOptimizerNoAnchor.set_anchor() is a no-op.
        for vi_opt in self._vi_optimizers.values():
            vi_opt.set_anchor()


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm & Config
# ─────────────────────────────────────────────────────────────────────────────

class IppoVINoAnchor(IppoVI):
    """IPPO with :class:`VIOptimizerNoAnchor` (gradient normalization only, no anchor).

    Optimizer difference is handled by :class:`VIOptimizerNoAnchorCallback`.
    """


@dataclass
class IppoVINoAnchorConfig(IppoVIConfig):
    @staticmethod
    def associated_class():
        return IppoVINoAnchor

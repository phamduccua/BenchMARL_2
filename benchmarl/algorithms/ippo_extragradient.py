#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""IPPO with Extragradient (ExtraAdam) actor optimizer.

The extragradient method (Korpelevich 1976) replaces each gradient step with
two steps:

    1. Extrapolation (look-ahead):
           θ_half = θ - lr_adam * Adam(∇L(θ))
    2. True update (from saved θ, using grad at θ_half):
           θ_new  = θ  - lr_adam * Adam(∇L(θ_half))

This is known to converge better in game-theoretic settings (multi-agent).

Usage::

    from benchmarl.algorithms.ippo_extragradient import (
        IppoExtragradientConfig,
        ExtraGradientCallback,
    )

    experiment = Experiment(
        ...
        algorithm_config=IppoExtragradientConfig(...),
        callbacks=[ExtraGradientCallback()],
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
# ExtraAdamOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class ExtraAdamOptimizer(optim.Optimizer):
    """Adam-based extragradient optimizer.

    ``step()`` performs the **extrapolation** sub-step using the gradients
    that were already computed by the preceding ``loss.backward()`` call:

    * Updates Adam moment estimates with grad(θ).
    * Saves θ  →  ``state["param_saved"]``.
    * Moves parameters to θ_half = θ − α·Adam(grad(θ)).

    After the caller recomputes the loss at θ_half and calls ``backward()``,
    ``update_from_current_grad()`` performs the **true update** sub-step:

    * Uses grad(θ_half) (now in ``p.grad``).
    * Restores θ from ``state["param_saved"]``.
    * Applies θ_new = θ − α·Adam(grad(θ_half)).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _adam_update(self, p: torch.Tensor, grad: torch.Tensor, group: dict, state: dict):
        """Compute Adam update vector (bias-corrected). Returns the update."""
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bc1 = 1 - beta1 ** state["step"]
        bc2 = 1 - beta2 ** state["step"]
        denom = (state["exp_avg_sq"].sqrt() / bc2 ** 0.5).add_(eps)
        return (state["exp_avg"] / bc1) / denom  # shape == p.shape

    # ── extrapolation step (called by BenchMARL's standard loop) ─────────────

    def step(self, closure=None):
        """Extrapolation step: uses the current ``p.grad``, saves θ, moves to θ_half."""
        if closure is not None:
            closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if wd:
                        grad = grad.add(p.data, alpha=wd)

                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                    state["step"] += 1
                    update = self._adam_update(p, grad, group, state)

                    state["param_saved"] = p.data.clone()
                    p.data.add_(update, alpha=-lr)          # θ_half

    # ── true-update step (called by ExtraGradientCallback) ───────────────────

    def update_from_current_grad(self):
        """True update: uses the current ``p.grad`` (at θ_half), restores θ, applies update."""
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if "param_saved" not in self.state[p]:
                        raise RuntimeError(
                            "ExtraAdamOptimizer: step() (extrapolation) must be called before "
                            "update_from_current_grad()."
                        )
                    grad = p.grad.data
                    if wd:
                        grad = grad.add(p.data, alpha=wd)

                    state = self.state[p]
                    # Compute Adam update at θ_half WITHOUT advancing step counter
                    ea  = beta1 * state["exp_avg"]    + (1 - beta1) * grad
                    ea2 = beta2 * state["exp_avg_sq"] + (1 - beta2) * grad * grad
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    denom = (ea2.sqrt() / bc2 ** 0.5).add_(eps)
                    update = (ea / bc1) / denom

                    # Restore θ and apply update from that point
                    p.data.copy_(state["param_saved"])
                    p.data.add_(update, alpha=-lr)          # θ_new

                    # Commit moment estimates
                    state["exp_avg"].copy_(ea)
                    state["exp_avg_sq"].copy_(ea2)


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────

class ExtraGradientCallback(Callback):
    """Replaces the ``loss_objective`` (actor) optimizer with
    :class:`ExtraAdamOptimizer` and performs the second gradient step
    (at θ_half) inside :meth:`on_train_step`.

    Flow per training iteration
    ---------------------------
    BenchMARL loop:

    1. ``losses[group](batch)``  → actor loss at **θ**
    2. ``loss.backward()``       → grad at **θ**
    3. ``ExtraAdamOptimizer.step()``  → save θ, move to **θ_half**
    4. ``optimizer.zero_grad()``

    ``ExtraGradientCallback.on_train_step``:

    5. Re-compute actor loss at **θ_half** (params are at θ_half)
    6. ``backward()``            → grad at **θ_half**
    7. ``update_from_current_grad()`` → restore **θ**, update to **θ_new**
    """

    def __init__(self):
        super().__init__()
        self._extra_optimizers: Dict[str, ExtraAdamOptimizer] = {}

    def on_setup(self):
        exp = self.experiment
        for group in exp.group_map.keys():
            if "loss_objective" not in exp.optimizers[group]:
                continue
            old_opt = exp.optimizers[group]["loss_objective"]
            extra_opt = ExtraAdamOptimizer(
                [{"params": pg["params"]} for pg in old_opt.param_groups],
                lr=exp.config.lr,
                betas=getattr(exp.algorithm_config, "extra_betas", (0.9, 0.999)),
                eps=getattr(exp.config, "adam_eps", 1e-8),
            )
            exp.optimizers[group]["loss_objective"] = extra_opt
            self._extra_optimizers[group] = extra_opt

    def on_train_step(self, batch: TensorDictBase, group: str) -> None:
        if group not in self._extra_optimizers:
            return None

        extra_opt = self._extra_optimizers[group]
        exp = self.experiment

        # ── 2nd forward/backward at θ_half ────────────────────────────────
        loss_vals = exp.losses[group](batch)
        actor_loss = loss_vals["loss_objective"]
        if "loss_entropy" in loss_vals.keys():
            actor_loss = actor_loss + loss_vals["loss_entropy"]

        extra_opt.zero_grad()
        actor_loss.backward()
        extra_opt.update_from_current_grad()
        extra_opt.zero_grad()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm & Config
# ─────────────────────────────────────────────────────────────────────────────

class IppoExtragradient(IppoVI):
    """IPPO with ExtraAdam actor optimizer (extragradient method).

    Identical to :class:`~benchmarl.algorithms.IppoVI` — the optimizer
    difference is handled entirely by :class:`ExtraGradientCallback`.
    """

    def __init__(self, extra_betas: tuple = (0.9, 0.999), **kwargs):
        # extra_betas is a config-only field consumed by ExtraGradientCallback.
        # We absorb it here so it doesn't reach Algorithm.__init__() which
        # would raise TypeError for an unexpected keyword argument.
        self.extra_betas = extra_betas
        super().__init__(**kwargs)


@dataclass
class IppoExtragradientConfig(IppoVIConfig):
    """Configuration for :class:`IppoExtragradient`.

    Extra parameters
    ----------------
    extra_betas : tuple
        Adam β₁, β₂ for the ExtraAdam actor optimizer.
    """

    extra_betas: tuple = (0.9, 0.999)

    @staticmethod
    def associated_class():
        return IppoExtragradient

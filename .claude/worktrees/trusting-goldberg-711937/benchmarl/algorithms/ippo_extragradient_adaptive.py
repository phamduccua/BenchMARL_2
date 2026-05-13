from dataclasses import dataclass
from typing import Dict

import torch
import torch.optim as optim
from tensordict import TensorDictBase

from benchmarl.algorithms.ippo_vi import IppoVI, IppoVIConfig
from benchmarl.experiment.callback import Callback


class AdaptiveExtraAdamOptimizer(optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        nu=0.95,
        rho=1e-4,
        min_scale=0.1,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, nu=nu, rho=rho, min_scale=min_scale)
        super().__init__(params, defaults)

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
                        state["current_lr"] = lr

                    state["step"] += 1
                    update = self._adam_update(p, grad, group, state)

                    state["param_saved"] = p.data.clone()
                    state["grad_saved"] = grad.clone()
                    state["update_saved"] = update.clone()

                    theta_half = p.data.clone()
                    theta_half.add_(
                        update,
                        alpha=-state["current_lr"]
                    )

                    state["theta_half"] = theta_half.clone()
                    p.data.copy_(theta_half)

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

                    state = self.state[p]

                    grad = p.grad.data
                    if wd:
                        grad = grad.add(
                            state["param_saved"],
                            alpha=wd
                        )

                    # Compute Adam update at θ_half WITHOUT advancing step counter
                    ea  = beta1 * state["exp_avg"]    + (1 - beta1) * grad
                    ea2 = beta2 * state["exp_avg_sq"] + (1 - beta2) * grad * grad
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    denom = (ea2.sqrt() / bc2 ** 0.5).add_(eps)
                    update = (ea / bc1) / denom

                    saved_grad = state["grad_saved"]
                    grad_dist = torch.norm(grad - saved_grad)

                    param_dist = torch.norm(
                        state["theta_half"] - state["param_saved"]
                    )

                    adaptive_lr = min(
                        group["nu"] * param_dist.item()
                        / (grad_dist.item() + eps),
                        state["current_lr"] + group["rho"]
                    )

                    # Clamp LR: không vượt quá 2× và không dưới min_scale× current_lr
                    adaptive_lr = max(
                        min(
                            adaptive_lr,
                            state["current_lr"] * 2.0
                        ),
                        state["current_lr"] * group["min_scale"]
                    )

                    # Restore original theta
                    p.data.copy_(state["param_saved"])

                    # Adaptive extragradient update
                    p.data.add_(update, alpha=-adaptive_lr)

                    state["current_lr"] = adaptive_lr

                    # Commit moment estimates
                    state["exp_avg"].copy_(ea)
                    state["exp_avg_sq"].copy_(ea2)


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveExtragradientCallback(Callback):

    def __init__(self):
        super().__init__()
        self._extra_optimizers: Dict[str, AdaptiveExtraAdamOptimizer] = {}

    def on_setup(self):
        exp = self.experiment
        for group in exp.group_map.keys():
            if "loss_objective" not in exp.optimizers[group]:
                continue
            old_opt = exp.optimizers[group]["loss_objective"]
            extra_opt = AdaptiveExtraAdamOptimizer(
                [{"params": pg["params"]} for pg in old_opt.param_groups],
                lr=exp.config.lr,
                betas=getattr(exp.algorithm_config, "extra_betas", (0.9, 0.999)),
                eps=getattr(exp.config, "adam_eps", 1e-8),
                nu=getattr(exp.algorithm_config, "extra_nu", 0.95),
                rho=getattr(exp.algorithm_config, "extra_rho", 1e-4),
                min_scale=getattr(exp.algorithm_config, "extra_min_scale", 0.1),
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

class IppoAdaptiveExtragradient(IppoVI):

    def __init__(self, extra_betas: tuple = (0.9, 0.999), **kwargs):
        self.extra_betas = extra_betas
        super().__init__(**kwargs)


@dataclass
class IppoAdaptiveExtragradientConfig(IppoVIConfig):
    """Configuration for :class:`IppoAdaptiveExtragradient`.

    Extra parameters
    ----------------
    extra_betas : tuple
        Adam β₁, β₂ for the AdaptiveExtraAdam actor optimizer.
    extra_nu : float
        Scales the ratio param_dist / grad_dist to compute adaptive LR target.
    extra_rho : float
        Maximum LR increment per step (caps how fast LR can grow).
    """

    extra_betas: tuple = (0.9, 0.999)
    extra_nu: float = 0.95
    extra_rho: float = 1e-4
    extra_min_scale: float = 0.1

    @staticmethod
    def associated_class():
        return IppoAdaptiveExtragradient

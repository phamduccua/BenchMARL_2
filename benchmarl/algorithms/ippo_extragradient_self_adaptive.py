from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.optim as optim
from tensordict import TensorDictBase

from benchmarl.algorithms.ippo_vi import IppoVI, IppoVIConfig
from benchmarl.experiment.callback import Callback



class SelfAdaptiveExtraGradientOptimizer(optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 1.9,
        nu: float = 0.95,
        rho: float = 1e-4,
        min_scale: float = 0.1,
        min_lr_scale: float = 0.1,
        tau_max: Optional[float] = None,
        rho_decay_power: float = 0.0,
    ):
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay, 
            gamma=gamma, 
            nu=nu, 
            rho=rho, 
            min_scale=min_scale,
            min_lr_scale=min_lr_scale,
            tau_max=tau_max,
            rho_decay_power=rho_decay_power,
        )
        super().__init__(params, defaults)
        self.current_lr = lr
        self.adaptive_step = 0
        self.last_stats: Dict[str, float] = {}

    def _adam_update(self, p: torch.Tensor, grad: torch.Tensor, group: dict, state: dict):
        """Compute Adam update vector (bias-corrected). Returns the update."""
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bc1 = 1 - beta1 ** state["step"]
        bc2 = 1 - beta2 ** state["step"]
        denom = (state["exp_avg_sq"].sqrt() / bc2 ** 0.5).add_(eps)
        return (state["exp_avg"] / bc1) / denom

    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
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
                    state["extrapolation_step"] = self.adaptive_step

                    # u_k in the actual Adam-preconditioned parameter geometry.
                    state["update_saved"] = update.clone()

                    theta_half = p.data.clone()

                    theta_half.add_(
                        update,
                        alpha=-self.current_lr
                    )

                    state["theta_half"] = theta_half.clone()
                    p.data.copy_(theta_half)
        return loss

    def update_from_current_grad(self):
        """True update: uses the current ``p.grad`` (at θ_half), restores θ, applies update."""
        with torch.no_grad():
            if not self.param_groups:
                return

            opts = self.param_groups[0]
            eps = opts["eps"]
            gamma = opts["gamma"]
            nu = opts["nu"]
            rho = opts["rho"]
            min_scale = opts["min_scale"]
            min_lr_scale = opts["min_lr_scale"]
            tau_max = opts["tau_max"]
            rho_decay_power = opts["rho_decay_power"]

            current_lr = self.current_lr
            rho_k = (
                rho / float(self.adaptive_step + 1) ** rho_decay_power
                if rho_decay_power > 0.0
                else rho
            )

            updates = []
            param_dist_sq = 0.0
            grad_dist_sq = 0.0
            numerator = 0.0
            denominator = 0.0

            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        state = self.state[p]
                        if state.get("extrapolation_step") == self.adaptive_step:
                            p.data.copy_(state["param_saved"])
                        continue
                    if (
                        "param_saved" not in self.state[p]
                        or "theta_half" not in self.state[p]
                        or "update_saved" not in self.state[p]
                        or self.state[p].get("extrapolation_step") != self.adaptive_step
                    ):
                        raise RuntimeError(
                            "SelfAdaptiveExtraGradientOptimizer: step() (extrapolation) must be called before "
                            "update_from_current_grad()."
                        )

                    state = self.state[p]

                    # Raw gradient at theta_half.
                    grad = p.grad.data

                    if wd:
                        grad = grad.add(
                            state["param_saved"],
                            alpha=wd
                        )

                    true_grad = grad

                    # v_k: Adam-preconditioned direction at theta_half, without advancing step.
                    ea = beta1 * state["exp_avg"] + (1 - beta1) * true_grad
                    ea2 = beta2 * state["exp_avg_sq"] + (1 - beta2) * true_grad * true_grad
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    denom = (ea2.sqrt() / bc2 ** 0.5).add_(eps)
                    true_update = (ea / bc1) / denom

                    # u_k: direction actually used to form theta_half.
                    saved_update = state["update_saved"]

                    x_minus_y = (
                        state["param_saved"]
                        - state["theta_half"]
                    )
                    update_diff = saved_update - true_update
                    d_k = x_minus_y - current_lr * update_diff

                    param_dist_sq += torch.sum(x_minus_y * x_minus_y).item()
                    grad_dist_sq += torch.sum(update_diff * update_diff).item()
                    numerator += torch.sum(x_minus_y * d_k).item()
                    denominator += torch.sum(d_k * d_k).item()
                    updates.append((p, state, ea, ea2, true_update))

            if not updates:
                return

            param_dist = param_dist_sq ** 0.5
            grad_dist = grad_dist_sq ** 0.5

            # lambda_{k+1}, computed once for the whole actor parameter vector.
            adaptive_lr = min(
                nu * param_dist / (grad_dist + eps),
                current_lr + rho_k,
            )
            base_lr = opts["lr"]
            min_lr = base_lr * min_lr_scale
            adaptive_lr = max(
                min(adaptive_lr, current_lr * 2.0),
                max(current_lr * min_scale, min_lr),
            )

            tau_k = gamma * abs(numerator) / (denominator + eps)
            tau_k = max(float(tau_k), 0.0)
            if tau_max is not None:
                tau_k = min(tau_k, tau_max)

            final_step = tau_k * current_lr

            for p, state, ea, ea2, true_update in updates:
                # x_{k+1} = x_k - tau_k lambda_k v_k
                p.data.copy_(state["param_saved"])
                p.data.add_(true_update, alpha=-final_step)
                state["exp_avg"].copy_(ea)
                state["exp_avg_sq"].copy_(ea2)

            self.current_lr = adaptive_lr
            self.adaptive_step += 1
            self.last_stats = {
                "lambda": current_lr,
                "lambda_next": adaptive_lr,
                "min_lr": min_lr,
                "rho_k": rho_k,
                "tau": tau_k,
                "final_step": final_step,
                "effective_step_ratio": final_step / base_lr if base_lr else 0.0,
                "param_dist": param_dist,
                "grad_dist": grad_dist,
            }


class SelfAdaptiveExtraGradientCallback(Callback):

    def __init__(self):
        super().__init__()
        self._extra_optimizers: Dict[str, SelfAdaptiveExtraGradientOptimizer] = {}

    def on_setup(self):
        exp = self.experiment
        for group in exp.group_map.keys():
            if "loss_objective" not in exp.optimizers[group]:
                continue
            old_opt = exp.optimizers[group]["loss_objective"]
            extra_opt = SelfAdaptiveExtraGradientOptimizer(
                [{"params": pg["params"]} for pg in old_opt.param_groups],
                lr=exp.config.lr,
                betas=getattr(exp.algorithm_config, "extra_betas", (0.9, 0.999)),
                eps=getattr(exp.config, "adam_eps", 1e-8),
                gamma=getattr(
                    exp.algorithm_config,
                    "extra_gamma",
                    1.9
                ),

                nu=getattr(
                    exp.algorithm_config,
                    "extra_nu",
                    0.95
                ),

                rho=getattr(
                    exp.algorithm_config,
                    "extra_rho",
                    1e-4
                ),

                min_scale=getattr(
                    exp.algorithm_config,
                    "extra_min_scale",
                    0.1
                ),
                min_lr_scale=getattr(
                    exp.algorithm_config,
                    "extra_min_lr_scale",
                    0.1
                ),
                tau_max=getattr(
                    exp.algorithm_config,
                    "extra_tau_max",
                    None
                ),
                rho_decay_power=getattr(
                    exp.algorithm_config,
                    "extra_rho_decay_power",
                    0.0
                ),
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

        # true update:
        # x_{k+1} = x_k - tau_k lambda_k v_k
        extra_opt.update_from_current_grad()

        log_interval = getattr(exp.algorithm_config, "extra_log_interval", 0)
        if log_interval and extra_opt.last_stats and exp.n_iters_performed % log_interval == 0:
            exp.logger.log(
                {
                    f"train/self_adaptive/{group}/{key}": value
                    for key, value in extra_opt.last_stats.items()
                },
                step=exp.n_iters_performed,
            )

        extra_opt.zero_grad()
        return None

class IppoSelfAdaptiveExtraGradient(IppoVI):

    def __init__(
        self,
        extra_betas: tuple = (0.9, 0.999),
        extra_gamma: float = 1.9,
        extra_nu: float = 0.95,
        extra_rho: float = 1e-4,
        extra_min_scale: float = 0.1,
        extra_min_lr_scale: float = 0.1,
        extra_tau_max: Optional[float] = None,
        extra_rho_decay_power: float = 0.0,
        extra_log_interval: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.extra_betas = extra_betas
        self.extra_gamma = extra_gamma
        self.extra_nu = extra_nu
        self.extra_rho = extra_rho
        self.extra_min_scale = extra_min_scale
        self.extra_min_lr_scale = extra_min_lr_scale
        self.extra_tau_max = extra_tau_max
        self.extra_rho_decay_power = extra_rho_decay_power
        self.extra_log_interval = extra_log_interval


@dataclass
class IppoSelfAdaptiveExtraGradientConfig(IppoVIConfig):
    extra_betas: tuple = (0.9, 0.999)
    extra_gamma: float = 1.9
    extra_nu: float = 0.95
    extra_rho: float = 1e-4
    extra_min_scale: float = 0.1
    extra_min_lr_scale: float = 0.1
    extra_tau_max: Optional[float] = None
    extra_rho_decay_power: float = 0.0
    extra_log_interval: int = 0

    @staticmethod
    def associated_class():
        return IppoSelfAdaptiveExtraGradient

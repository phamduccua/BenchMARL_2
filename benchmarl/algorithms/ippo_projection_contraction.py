from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from tensordict import TensorDictBase

from benchmarl.algorithms.ippo_vi import IppoVI, IppoVIConfig
from benchmarl.experiment.callback import Callback


# ============================================================
# Projection-Contraction Extra-Gradient Adam
# Variational Inequality MARL Optimizer
# ============================================================

class PCExtraGradientAdam(optim.Optimizer):

    def __init__(
        self,
        params,

        # Adam
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,

        # Projection-Contraction
        pc_beta: float = 1.0,
        pc_theta: float = 0.95,

        # stability
        pc_lambda_min: float = 1e-6,
        pc_lambda_max: float = 1.0,

        pc_beta_min: float = 0.05,
        pc_beta_max: float = 2.0,

        grad_clip: float = 10.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

        self.current_lr = lr

        self.pc_beta = pc_beta
        self.pc_theta = pc_theta

        self.pc_lambda_min = pc_lambda_min
        self.pc_lambda_max = pc_lambda_max

        self.pc_beta_min = pc_beta_min
        self.pc_beta_max = pc_beta_max

        self.grad_clip = grad_clip

        self.last_stats = {}

    # ============================================================
    # Utilities
    # ============================================================

    def _get_all_params(self) -> List[torch.Tensor]:
        params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
        return params

    def _clip_grad_by_norm(self, grad: torch.Tensor) -> torch.Tensor:
        """Clip a detached gradient tensor by global norm."""
        grad_norm = grad.norm()
        if grad_norm > self.grad_clip:
            grad = grad * (self.grad_clip / (grad_norm + 1e-8))
        return grad

    # ============================================================
    # Adam preconditioned direction (stateless read)
    # Does NOT mutate state — only reads current moments.
    # ============================================================

    def _adam_direction(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        step: int,
        group: dict,
    ) -> torch.Tensor:
        """
        Compute the Adam-preconditioned direction given pre-computed
        moment estimates and a step counter.  Pure function — no side effects.
        """
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        new_exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        new_exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step

        denom = (new_exp_avg_sq.sqrt() / (bc2 ** 0.5)).add_(eps)
        direction = (new_exp_avg / bc1) / denom

        return direction, new_exp_avg, new_exp_avg_sq

    # ============================================================
    # Initialise state for a parameter if not yet done
    # ============================================================

    def _init_state(self, p: torch.Tensor) -> dict:
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p.data)
            state["exp_avg_sq"] = torch.zeros_like(p.data)
        return state

    # ============================================================
    # STEP 1: extrapolation
    # theta_half = theta_k - lambda * F_adam(theta_k)
    #
    # Saves: theta_k, F_theta_k (Adam direction at theta_k),
    #        exp_avg / exp_avg_sq AFTER the first gradient.
    # ============================================================

    @torch.no_grad()
    def extrapolation_step(self):
        for group in self.param_groups:
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach().clone()

                if wd > 0:
                    grad = grad.add(p.data, alpha=wd)

                # Correct gradient clipping on the detached tensor
                grad = self._clip_grad_by_norm(grad)

                state = self._init_state(p)
                state["step"] += 1  # step 1 of 2

                direction, new_exp_avg, new_exp_avg_sq = self._adam_direction(
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group,
                )

                # Persist moments from the first gradient evaluation
                state["exp_avg"].copy_(new_exp_avg)
                state["exp_avg_sq"].copy_(new_exp_avg_sq)

                # Save anchor point and operator value
                state["theta_k"] = p.data.clone()
                state["F_theta_k"] = direction.clone()

                # Extrapolated point
                theta_half = p.data - self.current_lr * direction
                state["theta_half"] = theta_half.clone()

                p.data.copy_(theta_half)

    # ============================================================
    # STEP 2: projection-contraction update
    # theta_{k+1} = theta_k - beta_k * d_k
    # where d_k = (theta_k - theta_half) - lambda * (F(theta_k) - F(theta_half))
    # ============================================================

    @torch.no_grad()
    def pc_update(self):
        eps = 1e-8

        all_x_minus_y: List[torch.Tensor] = []
        all_operator_diff: List[torch.Tensor] = []
        all_dk: List[torch.Tensor] = []

        cache = []

        # --------------------------------------------------------
        # Collect per-parameter quantities
        # --------------------------------------------------------
        for group in self.param_groups:
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Guard: extrapolation_step must have been called first
                if "theta_k" not in state:
                    continue

                grad_half = p.grad.detach().clone()

                if wd > 0:
                    grad_half = grad_half.add(p.data, alpha=wd)

                grad_half = self._clip_grad_by_norm(grad_half)

                # NOTE: we do NOT increment step here — the second
                # gradient evaluation is part of the same iteration.
                # We reuse state["step"] (already incremented above)
                # to keep bias correction consistent.
                direction_half, exp_avg_new, exp_avg_sq_new = self._adam_direction(
                    grad_half,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group,
                )

                F_theta_k = state["F_theta_k"]

                # x - y  ≡  theta_k - theta_half
                x_minus_y = state["theta_k"] - state["theta_half"]

                # ΔF ≡  F(theta_k) - F(theta_half)
                operator_diff = F_theta_k - direction_half

                # d_k = (theta_k - theta_half) - lambda * (F(theta_k) - F(theta_half))
                d_k = x_minus_y - self.current_lr * operator_diff

                all_x_minus_y.append(x_minus_y.reshape(-1))
                all_operator_diff.append(operator_diff.reshape(-1))
                all_dk.append(d_k.reshape(-1))

                cache.append((
                    p,
                    state,
                    exp_avg_new,
                    exp_avg_sq_new,
                    d_k,
                ))

        if not cache:
            return

        # --------------------------------------------------------
        # Global vector geometry
        # --------------------------------------------------------
        x_minus_y_vec = torch.cat(all_x_minus_y)
        operator_diff_vec = torch.cat(all_operator_diff)
        d_k_vec = torch.cat(all_dk)

        param_dist = torch.norm(x_minus_y_vec)
        grad_dist = torch.norm(operator_diff_vec)
        dk_norm_sq = torch.sum(d_k_vec * d_k_vec) + eps
        numerator = torch.sum(x_minus_y_vec * d_k_vec)

        # --------------------------------------------------------
        # Adaptive step-size: lambda_{k+1}
        # Lipschitz-style: lambda <= theta * ||x - y|| / ||ΔF||
        # --------------------------------------------------------
        if grad_dist.item() > eps:
            candidate_lr = (self.pc_theta * param_dist / (grad_dist + eps)).item()
        else:
            candidate_lr = self.current_lr

        adaptive_lr = float(
            max(self.pc_lambda_min,
                min(candidate_lr, self.pc_lambda_max, self.current_lr))
        )

        # --------------------------------------------------------
        # Contraction coefficient: beta_k
        # beta_k = pc_beta * <x-y, d_k> / ||d_k||^2
        # Clipped to [pc_beta_min, pc_beta_max] for stability
        # --------------------------------------------------------
        beta_k = float(
            max(self.pc_beta_min,
                min(self.pc_beta * numerator / dk_norm_sq, self.pc_beta_max))
        )

        # --------------------------------------------------------
        # Apply update: theta_{k+1} = theta_k - beta_k * d_k
        # --------------------------------------------------------
        for (p, state, exp_avg_new, exp_avg_sq_new, d_k) in cache:

            # Restore to anchor theta_k, then apply contraction
            p.data.copy_(state["theta_k"])
            p.data.add_(d_k, alpha=-beta_k)  # subtract beta_k * d_k

            # Commit updated Adam moments
            state["exp_avg"].copy_(exp_avg_new)
            state["exp_avg_sq"].copy_(exp_avg_sq_new)

            # Clean up per-iteration temporaries
            del state["theta_k"]
            del state["theta_half"]
            del state["F_theta_k"]

        self.current_lr = adaptive_lr

        self.last_stats = {
            "lambda_k": self.current_lr,
            "beta_k": beta_k,
            "final_step": beta_k * self.current_lr,
            "param_dist": param_dist.item(),
            "grad_dist": grad_dist.item(),
            "dk_norm": torch.norm(d_k_vec).item(),
        }


# ============================================================
# Callback
# ============================================================

class ProjectionContractionCallback(Callback):

    def __init__(self):
        super().__init__()
        self.optimizers: Dict[str, PCExtraGradientAdam] = {}

    def on_setup(self):
        exp = self.experiment

        for group in exp.group_map.keys():

            if "loss_objective" not in exp.optimizers[group]:
                continue

            old_opt = exp.optimizers[group]["loss_objective"]

            algo_cfg = exp.algorithm_config

            new_opt = PCExtraGradientAdam(
                [{"params": pg["params"]} for pg in old_opt.param_groups],
                lr=exp.config.lr,
                betas=getattr(algo_cfg, "extra_betas", (0.9, 0.999)),
                eps=getattr(exp.config, "adam_eps", 1e-8),
                weight_decay=getattr(exp.config, "weight_decay", 0.0),
                pc_beta=getattr(algo_cfg, "pc_beta", 1.0),
                pc_theta=getattr(algo_cfg, "pc_theta", 0.95),
                pc_lambda_min=getattr(algo_cfg, "pc_lambda_min", 1e-6),
                pc_lambda_max=getattr(algo_cfg, "pc_lambda_max", 1.0),
                pc_beta_min=getattr(algo_cfg, "pc_beta_min", 0.05),
                pc_beta_max=getattr(algo_cfg, "pc_beta_max", 2.0),
                grad_clip=getattr(algo_cfg, "grad_clip", 10.0),
            )

            exp.optimizers[group]["loss_objective"] = new_opt
            self.optimizers[group] = new_opt

    # ============================================================
    # Main MARL training hook — two-step VI update
    # ============================================================

    def on_train_step(
        self,
        batch: TensorDictBase,
        group: str,
    ):
        if group not in self.optimizers:
            return

        optimizer = self.optimizers[group]
        exp = self.experiment

        # --------------------------------------------------------
        # STEP 1 — gradient at theta_k, extrapolate to theta_half
        # --------------------------------------------------------
        optimizer.zero_grad()

        loss_vals = exp.losses[group](batch)

        actor_loss = loss_vals["loss_objective"]

        if "loss_entropy" in loss_vals:
            actor_loss = actor_loss + loss_vals["loss_entropy"]

        # Critic loss (was missing in original code)
        if "loss_critic" in loss_vals:
            actor_loss = actor_loss + loss_vals["loss_critic"]

        actor_loss.backward()

        optimizer.extrapolation_step()

        # --------------------------------------------------------
        # STEP 2 — gradient at theta_half, PC-VI update
        # --------------------------------------------------------
        optimizer.zero_grad()

        loss_vals_half = exp.losses[group](batch)

        actor_loss_half = loss_vals_half["loss_objective"]

        if "loss_entropy" in loss_vals_half:
            actor_loss_half = actor_loss_half + loss_vals_half["loss_entropy"]

        if "loss_critic" in loss_vals_half:
            actor_loss_half = actor_loss_half + loss_vals_half["loss_critic"]

        actor_loss_half.backward()

        optimizer.pc_update()

        optimizer.zero_grad()

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------
        log_interval = getattr(exp.algorithm_config, "pc_log_interval", 0)

        if log_interval > 0 and exp.n_iters_performed % log_interval == 0:
            exp.logger.log(
                {
                    f"train/pc_vi/{group}/{k}": v
                    for k, v in optimizer.last_stats.items()
                },
                step=exp.n_iters_performed,
            )


# ============================================================
# Algorithm
# ============================================================

class IppoPCVI(IppoVI):

    def __init__(
        self,

        # Adam
        extra_betas: Tuple[float, float] = (0.9, 0.999),

        # Projection-Contraction
        pc_beta: float = 1.0,
        pc_theta: float = 0.95,

        pc_lambda_min: float = 1e-6,
        pc_lambda_max: float = 1.0,

        pc_beta_min: float = 0.05,
        pc_beta_max: float = 2.0,

        grad_clip: float = 10.0,

        pc_log_interval: int = 0,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.extra_betas = extra_betas

        self.pc_beta = pc_beta
        self.pc_theta = pc_theta

        self.pc_lambda_min = pc_lambda_min
        self.pc_lambda_max = pc_lambda_max

        self.pc_beta_min = pc_beta_min
        self.pc_beta_max = pc_beta_max

        self.grad_clip = grad_clip

        self.pc_log_interval = pc_log_interval


# ============================================================
# Config
# ============================================================

@dataclass
class IppoPCVIConfig(IppoVIConfig):

    extra_betas: tuple = (0.9, 0.999)

    pc_beta: float = 1.0
    pc_theta: float = 0.95

    pc_lambda_min: float = 1e-6
    pc_lambda_max: float = 1.0

    pc_beta_min: float = 0.05
    pc_beta_max: float = 2.0

    grad_clip: float = 10.0

    pc_log_interval: int = 0

    @staticmethod
    def associated_class():
        return IppoPCVI
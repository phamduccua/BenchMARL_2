from dataclasses import dataclass
from typing import Dict, List

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
        betas=(0.9, 0.999),
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
    # Flatten utilities
    # ============================================================

    def _flatten_tensor_list(self, xs: List[torch.Tensor]):

        if len(xs) == 0:
            return torch.tensor([])

        return torch.cat([x.reshape(-1) for x in xs])

    def _get_all_params(self):

        params = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)

        return params

    # ============================================================
    # Adam preconditioned direction
    # ============================================================

    def _compute_adam_direction(
        self,
        p,
        grad,
        state,
        group,
    ):

        beta1, beta2 = group["betas"]
        eps = group["eps"]

        exp_avg = (
            beta1 * state["exp_avg"]
            + (1 - beta1) * grad
        )

        exp_avg_sq = (
            beta2 * state["exp_avg_sq"]
            + (1 - beta2) * grad * grad
        )

        bc1 = 1 - beta1 ** state["step"]
        bc2 = 1 - beta2 ** state["step"]

        denom = (
            exp_avg_sq.sqrt() / (bc2 ** 0.5)
        ).add_(eps)

        direction = (exp_avg / bc1) / denom

        return direction, exp_avg, exp_avg_sq

    # ============================================================
    # STEP 1:
    # theta_half = theta - lambda * F(theta)
    # ============================================================

    @torch.no_grad()
    def extrapolation_step(self):

        params = self._get_all_params()

        for group in self.param_groups:

            wd = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if wd > 0:
                    grad = grad.add(
                        p.data,
                        alpha=wd
                    )

                torch.nn.utils.clip_grad_norm_(
                    [p],
                    self.grad_clip
                )

                state = self.state[p]

                if len(state) == 0:

                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                direction, _, _ = self._compute_adam_direction(
                    p,
                    grad,
                    state,
                    group,
                )

                # save theta_k
                state["theta_k"] = p.data.clone()

                # save F(theta_k)
                state["F_theta_k"] = direction.clone()

                # theta_half
                theta_half = (
                    p.data
                    - self.current_lr * direction
                )

                state["theta_half"] = theta_half.clone()

                p.data.copy_(theta_half)

    # ============================================================
    # STEP 2:
    # true projection-contraction update
    # ============================================================

    @torch.no_grad()
    def pc_update(self):

        eps = 1e-8

        all_x_minus_y = []
        all_operator_diff = []
        all_dk = []

        cache = []

        # ========================================================
        # Build full game operator vector
        # ========================================================

        for group in self.param_groups:

            wd = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if wd > 0:
                    grad = grad.add(
                        p.data,
                        alpha=wd
                    )

                state = self.state[p]

                direction_half, exp_avg_new, exp_avg_sq_new = \
                    self._compute_adam_direction(
                        p,
                        grad,
                        state,
                        group,
                    )

                F_theta_k = state["F_theta_k"]

                x_minus_y = (
                    state["theta_k"]
                    - state["theta_half"]
                )

                operator_diff = (
                    F_theta_k
                    - direction_half
                )

                # d_k
                d_k = (
                    x_minus_y
                    - self.current_lr * operator_diff
                )

                all_x_minus_y.append(
                    x_minus_y.reshape(-1)
                )

                all_operator_diff.append(
                    operator_diff.reshape(-1)
                )

                all_dk.append(
                    d_k.reshape(-1)
                )

                cache.append(
                    (
                        p,
                        state,
                        direction_half,
                        exp_avg_new,
                        exp_avg_sq_new,
                        d_k,
                    )
                )

        if len(cache) == 0:
            return

        # ========================================================
        # Global vector geometry
        # ========================================================

        x_minus_y_vec = torch.cat(all_x_minus_y)

        operator_diff_vec = torch.cat(all_operator_diff)

        d_k_vec = torch.cat(all_dk)

        param_dist = torch.norm(x_minus_y_vec)

        grad_dist = torch.norm(operator_diff_vec)

        dk_norm_sq = (
            torch.sum(d_k_vec * d_k_vec)
            + eps
        )

        numerator = torch.sum(
            x_minus_y_vec * d_k_vec
        )

        # ========================================================
        # lambda_{k+1}
        # ========================================================

        candidate_lr = (
            self.pc_theta
            * param_dist
            / (grad_dist + eps)
        )

        adaptive_lr = min(
            candidate_lr.item(),
            self.current_lr
        )

        adaptive_lr = max(
            adaptive_lr,
            self.pc_lambda_min
        )

        adaptive_lr = min(
            adaptive_lr,
            self.pc_lambda_max
        )

        # ========================================================
        # beta_k
        # ========================================================

        beta_k = (
            self.pc_beta
            * numerator
            / dk_norm_sq
        )

        beta_k = beta_k.item()

        beta_k = max(
            self.pc_beta_min,
            min(beta_k, self.pc_beta_max)
        )

        # ========================================================
        # final VI projection-contraction update
        # ========================================================

        final_step = beta_k * self.current_lr

        for (
            p,
            state,
            direction_half,
            exp_avg_new,
            exp_avg_sq_new,
            d_k,
        ) in cache:

            p.data.copy_(state["theta_k"])

            # theta_{k+1}
            p.data.add_(
                d_k,
                alpha=-beta_k
            )

            # update Adam state
            state["exp_avg"].copy_(exp_avg_new)
            state["exp_avg_sq"].copy_(exp_avg_sq_new)

            del state["theta_k"]
            del state["theta_half"]
            del state["F_theta_k"]

        self.current_lr = adaptive_lr

        self.last_stats = {
            "lambda_k": self.current_lr,
            "beta_k": beta_k,
            "final_step": final_step,
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

        self.optimizers: Dict[
            str,
            PCExtraGradientAdam
        ] = {}

    def on_setup(self):

        exp = self.experiment

        for group in exp.group_map.keys():

            if "loss_objective" not in exp.optimizers[group]:
                continue

            old_opt = exp.optimizers[group]["loss_objective"]

            new_opt = PCExtraGradientAdam(
                [{"params": pg["params"]} for pg in old_opt.param_groups],

                lr=exp.config.lr,

                betas=getattr(
                    exp.algorithm_config,
                    "extra_betas",
                    (0.9, 0.999),
                ),

                eps=getattr(
                    exp.config,
                    "adam_eps",
                    1e-8,
                ),

                weight_decay=getattr(
                    exp.config,
                    "weight_decay",
                    0.0,
                ),

                pc_beta=getattr(
                    exp.algorithm_config,
                    "pc_beta",
                    1.0,
                ),

                pc_theta=getattr(
                    exp.algorithm_config,
                    "pc_theta",
                    0.95,
                ),

                pc_lambda_min=getattr(
                    exp.algorithm_config,
                    "pc_lambda_min",
                    1e-6,
                ),

                pc_lambda_max=getattr(
                    exp.algorithm_config,
                    "pc_lambda_max",
                    1.0,
                ),

                pc_beta_min=getattr(
                    exp.algorithm_config,
                    "pc_beta_min",
                    0.05,
                ),

                pc_beta_max=getattr(
                    exp.algorithm_config,
                    "pc_beta_max",
                    2.0,
                ),

                grad_clip=getattr(
                    exp.algorithm_config,
                    "grad_clip",
                    10.0,
                ),
            )

            exp.optimizers[group]["loss_objective"] = new_opt

            self.optimizers[group] = new_opt

    # ============================================================
    # main MARL training
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

        # ========================================================
        # STEP 1
        # gradient at theta_k
        # ========================================================

        optimizer.zero_grad()

        loss_vals = exp.losses[group](batch)

        actor_loss = loss_vals["loss_objective"]

        if "loss_entropy" in loss_vals.keys():
            actor_loss = (
                actor_loss
                + loss_vals["loss_entropy"]
            )

        actor_loss.backward()

        # extrapolation
        optimizer.extrapolation_step()

        # ========================================================
        # STEP 2
        # gradient at theta_half
        # ========================================================

        optimizer.zero_grad()

        loss_vals_half = exp.losses[group](batch)

        actor_loss_half = loss_vals_half["loss_objective"]

        if "loss_entropy" in loss_vals_half.keys():
            actor_loss_half = (
                actor_loss_half
                + loss_vals_half["loss_entropy"]
            )

        actor_loss_half.backward()

        # projection-contraction VI update
        optimizer.pc_update()

        optimizer.zero_grad()

        # ========================================================
        # logging
        # ========================================================

        log_interval = getattr(
            exp.algorithm_config,
            "pc_log_interval",
            0,
        )

        if (
            log_interval > 0
            and exp.n_iters_performed % log_interval == 0
        ):

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
        extra_betas=(0.9, 0.999),

        # Projection-Contraction
        pc_beta=1.0,
        pc_theta=0.95,

        pc_lambda_min=1e-6,
        pc_lambda_max=1.0,

        pc_beta_min=0.05,
        pc_beta_max=2.0,

        grad_clip=10.0,

        pc_log_interval=0,

        **kwargs
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
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.optim as optim
from benchmarl.algorithms.ippo_vi import IppoVI, IppoVIConfig
from benchmarl.experiment.callback import Callback
from dataclasses import dataclass
from tensordict import TensorDictBase


# ============================================================
# Projection-Contraction Optimizer (Algorithm 2 — Chuẩn)
#
# Mỗi iteration gồm 2 bước:
#   Step 1 (extrapolation_step):
#       v_k = u_k − λ_k · AdamDir(grad(u_k))
#       Lưu: u_k, F_u_k = AdamDir(grad(u_k)), v_k
#
#   Step 2 (pc_update):
#       Tính F_v_k = AdamDir(grad(v_k))
#       d_k = λ_k · F_v_k       (vì x − y = λ_k·F(x), nên d_k = λ_k·F(y))
#       β_k = <u_k − v_k, d_k> / ||d_k||²   (chỉ dùng khi > 0)
#       u_{k+1} = u_k − β_k · d_k
#
#   Adaptive step-size cho bước tiếp theo:
#       λ_{k+1} = clip( θ · ||u_k − v_k|| / ||F(u_k) − F(v_k)|| )
# ============================================================

class PCOptimizer(optim.Optimizer):
    def __init__(
        self,
        params,
        # Adam
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        # PC Algorithm 2 params
        pc_beta=1.0,      # β trong công thức beta_k
        pc_theta=0.95,    # θ trong adaptive step-size
        pc_gamma=0.0,     # fallback beta_k khi numerator ≤ 0
        # Clipping
        grad_clip=10.0,
        lambda_min=1e-6,
        lambda_max=5e-4,
        lambda_growth=2.0,
        lambda_rho=1e-4,
        beta_min=0.0,
        beta_max=1.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

        self.current_lr = lr        # λ_k (adaptive step-size)
        self.initial_lr = lr
        self.pc_beta = pc_beta
        self.pc_theta = pc_theta
        self.pc_gamma = pc_gamma
        self.grad_clip = grad_clip
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_growth = lambda_growth
        self.lambda_rho = lambda_rho
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.iteration = 0
        self._cache = []            # lưu thông tin từ extrapolation_step
        self.last_stats = {}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _collect_grads(self):
        """Thu thập gradient hiện tại và clip global norm."""
        grads = {}
        total_norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().clone()
                grads[p] = g
                total_norm_sq += g.norm().item() ** 2

        total_norm = math.sqrt(total_norm_sq)
        if total_norm > self.grad_clip:
            coef = self.grad_clip / (total_norm + 1e-6)
            for p in grads:
                grads[p].mul_(coef)
        return grads

    def _adam_direction(self, grad, exp_avg, exp_avg_sq, step, beta1, beta2, eps):
        """Tính Adam update direction (bias-corrected), KHÔNG cập nhật moments."""
        m = beta1 * exp_avg + (1.0 - beta1) * grad
        v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        direction = (m / bc1) / (v.sqrt() / math.sqrt(bc2) + eps)
        return direction, m, v

    def _init_state(self, p):
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p.data)
            state["exp_avg_sq"] = torch.zeros_like(p.data)
        return state

    # ── Step 1: Extrapolation ─────────────────────────────────────────────────

    @torch.no_grad()
    def extrapolation_step(self):
        """
        Tính v_k = u_k − λ_k · AdamDir(grad(u_k)).
        Lưu u_k, F(u_k), v_k vào cache.
        """
        grads = self._collect_grads()
        self._cache = []

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p not in grads:
                    continue

                state = self._init_state(p)
                state["step"] += 1

                F_u_k, new_m, new_v = self._adam_direction(
                    grads[p],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    beta1, beta2, eps,
                )

                u_k = p.data.clone()
                v_k = u_k - self.current_lr * F_u_k

                # Lưu để dùng ở pc_update
                self._cache.append({
                    "p": p,
                    "state": state,
                    "u_k": u_k,
                    "v_k": v_k,
                    "F_u_k": F_u_k,
                    "new_m": new_m,
                    "new_v": new_v,
                    "beta1": beta1, "beta2": beta2, "eps": eps,
                })

                # Di chuyển tham số đến v_k để tính loss tại v_k
                p.data.copy_(v_k)

    # ── Step 2: PC Update ─────────────────────────────────────────────────────

    @torch.no_grad()
    def pc_update(self):
        """
        Tính PC update từ grad(v_k):
            d_k   = λ_k · F(v_k)
            β_k   = <u_k − v_k, d_k> / ||d_k||²
            u_{k+1} = u_k − β_k · d_k
        Rồi tính λ_{k+1} = θ · ||u_k − v_k|| / ||F(u_k) − F(v_k)||.
        """
        if not self._cache:
            return

        grads_vk = self._collect_grads()

        # --- Thu thập vectors toàn cục để tính β_k và λ_{k+1} ---
        all_xmy = []      # u_k - v_k  (= λ_k * F_u_k)
        all_fvk  = []     # F(v_k)
        all_fdiff = []    # F(u_k) - F(v_k)
        entry_data = []   # (entry, F_v_k per-param)

        for entry in self._cache:
            p = entry["p"]
            if p not in grads_vk:
                # Không có grad → bỏ qua param này
                continue

            state = entry["state"]
            beta1, beta2, eps = entry["beta1"], entry["beta2"], entry["eps"]

            # Tính F(v_k) với moments từ bước extrapolation (new_m, new_v)
            F_v_k, _, _ = self._adam_direction(
                grads_vk[p],
                entry["new_m"],
                entry["new_v"],
                state["step"],
                beta1, beta2, eps,
            )

            x_minus_y = entry["u_k"] - entry["v_k"]   # = λ_k * F_u_k
            f_diff = entry["F_u_k"] - F_v_k

            all_xmy.append(x_minus_y.reshape(-1))
            all_fvk.append(F_v_k.reshape(-1))
            all_fdiff.append(f_diff.reshape(-1))
            entry_data.append((entry, F_v_k))

        if not entry_data:
            # Không có gì để update, khôi phục tham số
            for entry in self._cache:
                entry["p"].data.copy_(entry["u_k"])
            self._cache = []
            return

        xmy_vec  = torch.cat(all_xmy)
        fvk_vec  = torch.cat(all_fvk)
        fdiff_vec = torch.cat(all_fdiff)

        # --- Tính β_k ---
        # d_k = λ_k * F(v_k)  →  ||d_k||² = λ_k² * ||F(v_k)||²
        # <u_k - v_k, d_k> = <λ_k*F(u_k), λ_k*F(v_k)> = λ_k² * <F(u_k), F(v_k)>
        # β_k = <u_k - v_k, d_k> / ||d_k||²
        #      = λ_k² * <F(u_k), F(v_k)> / (λ_k² * ||F(v_k)||²)
        #      = <F(u_k), F(v_k)> / ||F(v_k)||²
        # (λ_k triệt tiêu — β_k không phụ thuộc λ_k!)
        #
        # Công thức trực tiếp theo định nghĩa gốc:
        fvk_norm_sq = torch.sum(fvk_vec * fvk_vec).item()
        eps_safe = 1e-12

        if fvk_norm_sq > eps_safe:
            raw_beta = self.pc_beta * torch.dot(xmy_vec, self.current_lr * fvk_vec).item() / (
                self.current_lr ** 2 * fvk_norm_sq + eps_safe
            )
            # Tương đương: raw_beta = pc_beta * <F_u, F_v> / ||F_v||²
            if raw_beta <= 0:
                beta_k = self.pc_gamma
            else:
                beta_k = max(self.beta_min, min(self.beta_max, raw_beta))
        else:
            beta_k = self.pc_gamma

        # --- Tính λ_{k+1} (adaptive step-size) ---
        param_dist = torch.norm(xmy_vec).item()
        grad_dist  = torch.norm(fdiff_vec).item()

        if grad_dist > eps_safe:
            lambda_next = self.pc_theta * param_dist / (grad_dist + eps_safe)
        else:
            lambda_next = self.current_lr

        lambda_next = min(
            lambda_next,
            self.current_lr * self.lambda_growth,
            self.current_lr + self.lambda_rho,
        )
        lambda_next = max(self.lambda_min, min(self.lambda_max, lambda_next))

        # --- Áp dụng update cho từng tham số ---
        for (entry, F_v_k) in entry_data:
            p = entry["p"]
            state = entry["state"]

            # d_k = λ_k * F(v_k)
            d_k = self.current_lr * F_v_k

            # u_{k+1} = u_k - β_k * d_k
            u_next = entry["u_k"] - beta_k * d_k
            p.data.copy_(u_next)

            # Commit Adam moments (từ bước extrapolation tại u_k)
            state["exp_avg"].copy_(entry["new_m"])
            state["exp_avg_sq"].copy_(entry["new_v"])

        # Xử lý các params không có grad (khôi phục về u_k)
        updated_params = {entry["p"] for entry, _ in entry_data}
        for entry in self._cache:
            if entry["p"] not in updated_params:
                entry["p"].data.copy_(entry["u_k"])

        self._cache = []
        self.current_lr = lambda_next
        self.iteration += 1
        self.last_stats = {
            "lambda_k": self.current_lr,
            "beta_k": beta_k,
            "param_dist": param_dist,
            "grad_dist": grad_dist,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.extrapolation_step()
        return loss


# ============================================================
# Callback
# ============================================================

class ProjectionContractionCallback(Callback):
    def __init__(self):
        super().__init__()
        self.actor_optimizers: Dict[str, PCOptimizer] = {}

    def on_setup(self) -> None:
        exp = self.experiment
        algo_cfg = exp.algorithm_config

        for group in exp.group_map.keys():
            group_opts = exp.optimizers[group]

            # Khởi tạo Actor Optimizer
            if "loss_objective" in group_opts:
                old_opt = group_opts["loss_objective"]
                actor_opt = PCOptimizer(
                    params=[{"params": pg["params"]} for pg in old_opt.param_groups],
                    lr=exp.config.lr,
                    betas=getattr(algo_cfg, "extra_betas", (0.9, 0.999)),
                    eps=getattr(exp.config, "adam_eps", 1e-8),
                    pc_beta=getattr(algo_cfg, "pc_beta", 1.0),
                    pc_theta=getattr(algo_cfg, "pc_theta", 0.95),
                    pc_gamma=getattr(algo_cfg, "pc_gamma", 0.0),
                    grad_clip=getattr(algo_cfg, "grad_clip", 10.0),
                    lambda_min=getattr(algo_cfg, "pc_lambda_min", 1e-6),
                    lambda_max=getattr(algo_cfg, "pc_lambda_max", 5e-4),
                    lambda_growth=getattr(algo_cfg, "pc_lambda_growth", 2.0),
                    lambda_rho=getattr(algo_cfg, "pc_lambda_rho", 1e-4),
                    beta_min=getattr(algo_cfg, "pc_beta_min", 0.0),
                    beta_max=getattr(algo_cfg, "pc_beta_max", 1.0),
                )
                group_opts["loss_objective"] = actor_opt
                self.actor_optimizers[group] = actor_opt

    def on_train_step(self, batch: TensorDictBase, group: str) -> None:
        exp = self.experiment
        algo_cfg = exp.algorithm_config

        if group in self.actor_optimizers:
            actor_opt = self.actor_optimizers[group]
            loss_vals = exp.losses[group](batch.clone(True))
            actor_loss = loss_vals["loss_objective"]
            if "loss_entropy" in loss_vals:
                actor_loss = actor_loss + loss_vals["loss_entropy"]

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.pc_update()            # tham số về u_{k+1}
            actor_opt.zero_grad()

        # Logging
        if group in self.actor_optimizers:
            actor_opt = self.actor_optimizers[group]
            log_interval = getattr(algo_cfg, "pc_log_interval", 0)
            if log_interval > 0 and exp.n_iters_performed % log_interval == 0 and actor_opt.last_stats:
                stats = {
                    f"train/pc/{group}/{k}": v
                    for k, v in actor_opt.last_stats.items()
                }
                exp.logger.log(stats, step=exp.n_iters_performed)


# ============================================================
# Algorithm & Config
# ============================================================

class IppoPCVI(IppoVI):
    def __init__(
        self,
        extra_betas=(0.9, 0.999),
        pc_beta=1.0,
        pc_theta=0.95,
        pc_gamma=0.0,
        grad_clip=10.0,
        pc_lambda_min=1e-6,
        pc_lambda_max=5e-4,
        pc_lambda_growth=2.0,
        pc_lambda_rho=1e-4,
        pc_beta_min=0.0,
        pc_beta_max=1.0,
        pc_log_interval=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.extra_betas = extra_betas
        self.pc_beta = pc_beta
        self.pc_theta = pc_theta
        self.pc_gamma = pc_gamma
        self.grad_clip = grad_clip
        self.pc_lambda_min = pc_lambda_min
        self.pc_lambda_max = pc_lambda_max
        self.pc_lambda_growth = pc_lambda_growth
        self.pc_lambda_rho = pc_lambda_rho
        self.pc_beta_min = pc_beta_min
        self.pc_beta_max = pc_beta_max
        self.pc_log_interval = pc_log_interval


@dataclass
class IppoPCVIConfig(IppoVIConfig):
    extra_betas: Tuple[float, float] = (0.9, 0.999)
    pc_beta: float = 1.0
    pc_theta: float = 0.95
    pc_gamma: float = 0.0
    grad_clip: float = 10.0
    pc_lambda_min: float = 1e-6
    pc_lambda_max: float = 5e-4
    pc_lambda_growth: float = 2.0
    pc_lambda_rho: float = 1e-4
    pc_beta_min: float = 0.0
    pc_beta_max: float = 1.0
    pc_log_interval: int = 0

    @staticmethod
    def associated_class():
        return IppoPCVI


PCExtraGradientAdam = PCOptimizer

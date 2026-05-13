from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.optim as optim
from benchmarl.algorithms.ippo_vi import IppoVI, IppoVIConfig
from benchmarl.experiment.callback import Callback
from dataclasses import dataclass
from tensordict import TensorDictBase


# ============================================================
# Optimizer
# ============================================================

class PCExtraGradientAdam(optim.Optimizer):
    def __init__(
        self,
        params,
        # Adam
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        # Algorithm 2
        pc_beta=1.0,
        pc_theta=0.95,
        pc_gamma=0.1,
        # viscosity (nhỏ để tránh kìm hãm learning ban đầu)
        viscosity_scale=0.01,
        # EMA target
        target_tau=0.005,
        # stability
        grad_clip=10.0,
        lambda_min=1e-6,
        lambda_max=1.0,
        beta_min=1e-4,
        beta_max=2.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

        self.current_lr = lr
        self.pc_beta = pc_beta
        self.pc_theta = pc_theta
        self.pc_gamma = pc_gamma
        self.viscosity_scale = viscosity_scale
        self.target_tau = target_tau
        self.grad_clip = grad_clip
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.iteration = 0
        self.geometry = None
        self.last_stats = {}

    def viscosity_t(self):
        """
        Conditions: t_k -> 0, sum t_k = infinity.
        Warmup: không áp viscosity trong 200 iterations đầu để tránh kìm hãm learning.
        """
        warmup = 200
        if self.iteration < warmup:
            return 0.0
        return self.viscosity_scale / (self.iteration - warmup + 1)

    def _init_state(self, p):
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p.data)
            state["exp_avg_sq"] = torch.zeros_like(p.data)
            # EMA target actor
            state["target_param"] = p.data.clone()
        return state

    def _collect_grads(self):
        grads = {}
        total_norm = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().clone()
                grads[p] = g
                total_norm += g.norm().item() ** 2

        total_norm = math.sqrt(total_norm)
        clip_coef = self.grad_clip / (total_norm + 1e-6)

        if clip_coef < 1.0:
            for p in grads:
                grads[p] *= clip_coef

        return grads

    def _adam_direction(
        self, grad, exp_avg, exp_avg_sq, step, beta1, beta2, eps
    ):
        m = beta1 * exp_avg + (1.0 - beta1) * grad
        v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad

        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step

        direction = (m / bc1) / (v.sqrt() / math.sqrt(bc2) + eps)
        return direction, m, v

    @torch.no_grad()
    def extrapolation_step(self):
        grads = self._collect_grads()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p not in grads:
                    continue

                state = self._init_state(p)
                state["step"] += 1

                direction, new_m, new_v = self._adam_direction(
                    grads[p],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    beta1,
                    beta2,
                    eps,
                )

                # Lưu trạng thái u_k và F(u_k)
                state["u_k"] = p.data.clone()
                state["F_u_k"] = direction.clone()

                # Di chuyển sang v_k
                v_k = p.data - self.current_lr * direction
                state["v_k"] = v_k.clone()
                p.data.copy_(v_k)

                # Lưu tạm moments
                state["new_m"] = new_m
                state["new_v"] = new_v

    @torch.no_grad()
    def _collect_pc_geometry(self):
        grads_half = self._collect_grads()

        all_xmy = []
        all_opdiff = []
        all_dk = []
        cache = []

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                # Bỏ qua nếu không có đủ thông tin gradient hoặc chưa qua bước extrapolation
                if p not in grads_half or p not in self.state or "u_k" not in self.state[p]:
                    continue

                state = self.state[p]

                # FIX 1: Dùng moments ĐÃ CẬP NHẬT (new_m/new_v từ extrapolation_step)
                # thay vì moments cũ (exp_avg/exp_avg_sq chưa được commit)
                # để tính F(v_k) chính xác hơn.
                direction_half, _, _ = self._adam_direction(
                    grads_half[p],
                    state["new_m"],   # ✅ moments mới (đã tích lũy grad tại u_k)
                    state["new_v"],   # ✅ moments mới
                    state["step"],
                    beta1,
                    beta2,
                    eps,
                )

                x_minus_y = state["u_k"] - state["v_k"]
                op_diff = state["F_u_k"] - direction_half
                d_k = x_minus_y - self.current_lr * op_diff

                all_xmy.append(x_minus_y.reshape(-1))
                all_opdiff.append(op_diff.reshape(-1))
                all_dk.append(d_k.reshape(-1))

                cache.append((p, state, d_k))

        # Xử lý trường hợp an toàn nếu không thu thập được tensor nào
        if not all_xmy:
            dummy = torch.tensor([0.0], device=self.param_groups[0]["params"][0].device)
            self.geometry = {
                "param_dist": dummy, "grad_dist": dummy,
                "dk_norm_sq": dummy, "numerator": dummy, "cache": []
            }
            return

        xmy_vec = torch.cat(all_xmy)
        opdiff_vec = torch.cat(all_opdiff)
        dk_vec = torch.cat(all_dk)

        self.geometry = {
            "param_dist": torch.norm(xmy_vec),
            "grad_dist": torch.norm(opdiff_vec),
            "dk_norm_sq": torch.sum(dk_vec * dk_vec),
            "numerator": torch.sum(xmy_vec * dk_vec),
            "cache": cache,
        }

    def _compute_lambda(self):
        eps = 1e-12
        param_dist = self.geometry["param_dist"]
        grad_dist = self.geometry["grad_dist"]

        if grad_dist.item() > eps:
            candidate = (self.pc_theta * param_dist / (grad_dist + eps)).item()
        else:
            candidate = self.current_lr

        return max(self.lambda_min, min(self.lambda_max, candidate))

    def _compute_beta(self):
        eps = 1e-12
        numerator = self.geometry["numerator"]
        dk_norm_sq = self.geometry["dk_norm_sq"]

        if dk_norm_sq.item() < eps:
            return self.pc_gamma

        raw_beta = (self.pc_beta * numerator / (dk_norm_sq + eps)).item()
        if raw_beta <= 0:
            return self.pc_gamma

        return max(self.beta_min, min(self.beta_max, raw_beta))

    @torch.no_grad()
    def pc_update(self):
        self._collect_pc_geometry()

        # FIX 4: Tính lambda_k trước, áp dụng ngay trong bước update hiện tại
        # (không phải chờ đến iteration sau mới dùng)
        lambda_k = self._compute_lambda()
        beta_k = self._compute_beta()
        t_k = self.viscosity_t()

        for (p, state, d_k) in self.geometry["cache"]:
            # Khôi phục u_k
            p.data.copy_(state["u_k"])

            # w_k = u_k - β_k d_k  (projection-contraction step)
            w_k = p.data - beta_k * d_k
            target_param = state["target_param"]

            # Viscosity update: chỉ kéo về anchor khi t_k > 0 (sau warmup)
            if t_k > 0.0:
                u_next = t_k * target_param + (1.0 - t_k) * w_k
            else:
                u_next = w_k
            p.data.copy_(u_next)

            # EMA target update
            target_param.mul_(1.0 - self.target_tau).add_(p.data, alpha=self.target_tau)

            # Commit Adam moments
            state["exp_avg"].copy_(state["new_m"])
            state["exp_avg_sq"].copy_(state["new_v"])

            # Dọn dẹp cache để tránh rò rỉ bộ nhớ
            for k in ["u_k", "v_k", "F_u_k", "new_m", "new_v"]:
                state.pop(k, None)

        # Áp dụng lambda_k cho extrapolation step TIẾP THEO (step-size adaptive)
        self.current_lr = lambda_k
        self.iteration += 1
        self.last_stats = {"lambda_k": lambda_k, "beta_k": beta_k, "t_k": t_k}

    @torch.no_grad()
    def step(self, closure=None):
        # GHI ĐÈ: Để rỗng (pass) nhằm tránh BenchMARL mặc định gọi step() gây crash
        pass


# ============================================================
# Callback
# ============================================================

class ProjectionContractionCallback(Callback):
    def __init__(self):
        super().__init__()
        self.actor_optimizers: Dict[str, PCExtraGradientAdam] = {}
        self.critic_optimizers: Dict[str, optim.Adam] = {}

    def on_setup(self) -> None:
        exp = self.experiment
        algo_cfg = exp.algorithm_config

        for group in exp.group_map.keys():
            group_opts = exp.optimizers[group]

            # Khởi tạo Actor Optimizer
            if "loss_objective" in group_opts:
                old_opt = group_opts["loss_objective"]
                actor_opt = PCExtraGradientAdam(
                    params=[{"params": pg["params"]} for pg in old_opt.param_groups],
                    lr=exp.config.lr,
                    betas=getattr(algo_cfg, "extra_betas", (0.9, 0.999)),
                    eps=getattr(exp.config, "adam_eps", 1e-8),
                    pc_beta=getattr(algo_cfg, "pc_beta", 1.0),
                    pc_theta=getattr(algo_cfg, "pc_theta", 0.95),
                    pc_gamma=getattr(algo_cfg, "pc_gamma", 0.1),
                    viscosity_scale=getattr(algo_cfg, "viscosity_scale", 0.5),
                    target_tau=getattr(algo_cfg, "target_tau", 0.005),
                    grad_clip=getattr(algo_cfg, "grad_clip", 10.0),
                    lambda_min=getattr(algo_cfg, "pc_lambda_min", 1e-6),
                    lambda_max=getattr(algo_cfg, "pc_lambda_max", 1.0),
                    beta_min=getattr(algo_cfg, "pc_beta_min", 1e-4),
                    beta_max=getattr(algo_cfg, "pc_beta_max", 2.0),
                )
                group_opts["loss_objective"] = actor_opt
                self.actor_optimizers[group] = actor_opt

            # Critic Optimizer
            if "loss_critic" in group_opts:
                self.critic_optimizers[group] = group_opts["loss_critic"]

    def on_train_step(self, batch: TensorDictBase, group: str) -> None:
        exp = self.experiment
        algo_cfg = exp.algorithm_config

        if group in self.actor_optimizers:
            actor_opt = self.actor_optimizers[group]

            def _loss_fn(vals):
                l = vals["loss_objective"]
                if "loss_entropy" in vals:
                    l = l + vals["loss_entropy"]
                return l

            # FIX 3: Deep clone (True) để tránh data corruption do TorchRL
            # thực hiện in-place ops trên batch khi tính loss.
            # Shallow clone (False) khiến batch_step2 nhận batch đã bị sửa.
            batch_step1 = batch.clone(True)
            batch_step2 = batch.clone(True)

            # --- Bước 1: Extrapolation ---
            actor_opt.zero_grad()
            loss_step1 = _loss_fn(exp.losses[group](batch_step1))
            loss_step1.backward()
            actor_opt.extrapolation_step()

            # --- Bước 2: PC Update ---
            actor_opt.zero_grad()
            loss_step2 = _loss_fn(exp.losses[group](batch_step2))
            loss_step2.backward()
            actor_opt.pc_update()

            actor_opt.zero_grad()

        # Critic update (Dùng chuẩn Adam)
        if group in self.critic_optimizers:
            critic_opt = self.critic_optimizers[group]
            critic_opt.zero_grad()
            
            # FIX 3 (cont): Deep clone cho critic batch
            batch_critic = batch.clone(True)
            l_vals = exp.losses[group](batch_critic)
            
            if "loss_critic" in l_vals:
                l_vals["loss_critic"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in critic_opt.param_groups for p in pg["params"]],
                    max_norm=getattr(algo_cfg, "grad_clip", 10.0),
                )
                critic_opt.step()
            critic_opt.zero_grad()

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
        pc_gamma=0.1,
        viscosity_scale=0.01,
        target_tau=0.005,
        grad_clip=10.0,
        pc_lambda_min=1e-6,
        pc_lambda_max=1.0,
        pc_beta_min=1e-4,
        pc_beta_max=2.0,
        pc_log_interval=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.extra_betas = extra_betas
        self.pc_beta = pc_beta
        self.pc_theta = pc_theta
        self.pc_gamma = pc_gamma
        self.viscosity_scale = viscosity_scale
        self.target_tau = target_tau
        self.grad_clip = grad_clip
        self.pc_lambda_min = pc_lambda_min
        self.pc_lambda_max = pc_lambda_max
        self.pc_beta_min = pc_beta_min
        self.pc_beta_max = pc_beta_max
        self.pc_log_interval = pc_log_interval


@dataclass
class IppoPCVIConfig(IppoVIConfig):
    extra_betas: Tuple[float, float] = (0.9, 0.999)
    pc_beta: float = 1.0
    pc_theta: float = 0.95
    pc_gamma: float = 0.1
    viscosity_scale: float = 0.01
    target_tau: float = 0.005
    grad_clip: float = 10.0
    pc_lambda_min: float = 1e-6
    pc_lambda_max: float = 1.0
    pc_beta_min: float = 1e-4
    pc_beta_max: float = 2.0
    pc_log_interval: int = 0

    @staticmethod
    def associated_class():
        return IppoPCVI
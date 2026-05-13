"""
Projection-Contraction Extra-Gradient Adam
Algorithm 2 ONLY (Strong Convergence Version)

Mapped from SCSPMOS paper to MARL/IPPO.

Paper:
    v_k = u_k - λ_kF(u_k)

    d_k = u_k - v_k - λ_k(F(u_k)-F(v_k))

    β_k = β<u_k-v_k,d_k>/||d_k||²

    w_k = u_k - β_k d_k

    u_{k+1} = t_k g(u_k) + (1-t_k)w_k

MARL mapping:
    u_k      -> actor params
    F(u_k)   -> Adam-preconditioned policy gradient
    g(u_k)   -> EMA target actor
"""

from __future__ import annotations

import math
import torch
import torch.optim as optim

from typing import Dict, Optional, Tuple


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

        # viscosity
        viscosity_scale=0.5,

        # EMA target
        target_tau=0.005,

        # stability
        grad_clip=10.0,
        lambda_min=1e-6,
        lambda_max=1.0,
        beta_min=1e-4,
        beta_max=2.0,
    ):

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
        )

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

    # ========================================================
    # viscosity t_k
    # ========================================================

    def viscosity_t(self):

        """
        Conditions:
            t_k -> 0
            sum t_k = infinity
        """

        return self.viscosity_scale / (self.iteration + 1)

    # ========================================================
    # init state
    # ========================================================

    def _init_state(self, p):

        state = self.state[p]

        if len(state) == 0:

            state["step"] = 0

            state["exp_avg"] = torch.zeros_like(p.data)
            state["exp_avg_sq"] = torch.zeros_like(p.data)

            # EMA target actor
            state["target_param"] = p.data.clone()

        return state

    # ========================================================
    # collect grads
    # ========================================================

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

    # ========================================================
    # Adam-preconditioned operator F
    # ========================================================

    def _adam_direction(
        self,
        grad,
        exp_avg,
        exp_avg_sq,
        step,
        beta1,
        beta2,
        eps,
    ):

        m = beta1 * exp_avg + (1.0 - beta1) * grad

        v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad

        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step

        direction = (m / bc1) / (
            v.sqrt() / math.sqrt(bc2) + eps
        )

        return direction, m, v

    # ========================================================
    # Step 1
    # v_k = u_k - λ_kF(u_k)
    # ========================================================

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

                # save u_k
                state["u_k"] = p.data.clone()

                # save F(u_k)
                state["F_u_k"] = direction.clone()

                # move to v_k
                v_k = p.data - self.current_lr * direction

                state["v_k"] = v_k.clone()

                p.data.copy_(v_k)

                # temporary moments
                state["new_m"] = new_m
                state["new_v"] = new_v

    # ========================================================
    # collect geometry
    # ========================================================

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

                if p not in grads_half:
                    continue

                state = self.state[p]

                direction_half, _, _ = self._adam_direction(
                    grads_half[p],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    beta1,
                    beta2,
                    eps,
                )

                # u_k - v_k
                x_minus_y = state["u_k"] - state["v_k"]

                # F(u_k)-F(v_k)
                op_diff = state["F_u_k"] - direction_half

                # d_k
                d_k = x_minus_y - self.current_lr * op_diff

                all_xmy.append(x_minus_y.reshape(-1))
                all_opdiff.append(op_diff.reshape(-1))
                all_dk.append(d_k.reshape(-1))

                cache.append(
                    (
                        p,
                        state,
                        d_k,
                    )
                )

        xmy_vec = torch.cat(all_xmy)
        opdiff_vec = torch.cat(all_opdiff)
        dk_vec = torch.cat(all_dk)

        self.geometry = {
            "xmy_vec": xmy_vec,
            "opdiff_vec": opdiff_vec,
            "dk_vec": dk_vec,

            "param_dist": torch.norm(xmy_vec),
            "grad_dist": torch.norm(opdiff_vec),

            "dk_norm_sq": torch.sum(dk_vec * dk_vec),

            "numerator": torch.sum(xmy_vec * dk_vec),

            "cache": cache,
        }

    # ========================================================
    # lambda_k
    # ========================================================

    def _compute_lambda(self):

        eps = 1e-12

        param_dist = self.geometry["param_dist"]
        grad_dist = self.geometry["grad_dist"]

        if grad_dist.item() > eps:

            candidate = (
                self.pc_theta
                * param_dist
                / (grad_dist + eps)
            ).item()

        else:
            candidate = self.current_lr

        candidate = max(self.lambda_min, candidate)
        candidate = min(self.lambda_max, candidate)

        return candidate

    # ========================================================
    # beta_k
    # ========================================================

    def _compute_beta(self):

        eps = 1e-12

        numerator = self.geometry["numerator"]
        dk_norm_sq = self.geometry["dk_norm_sq"]

        if dk_norm_sq.item() < eps:
            return self.pc_gamma

        raw_beta = (
            self.pc_beta
            * numerator
            / (dk_norm_sq + eps)
        ).item()

        if raw_beta <= 0:
            return self.pc_gamma

        raw_beta = max(self.beta_min, raw_beta)
        raw_beta = min(self.beta_max, raw_beta)

        return raw_beta

    # ========================================================
    # Algorithm 2 update
    # ========================================================

    @torch.no_grad()
    def pc_update(self):

        self._collect_pc_geometry()

        lambda_k = self._compute_lambda()

        beta_k = self._compute_beta()

        t_k = self.viscosity_t()

        for (p, state, d_k) in self.geometry["cache"]:

            # restore u_k
            p.data.copy_(state["u_k"])

            # w_k = u_k - β_k d_k
            w_k = p.data - beta_k * d_k

            # g(u_k)
            target_param = state["target_param"]

            # Algorithm 2 viscosity:
            #
            # u_{k+1} = t_k g(u_k) + (1-t_k)w_k
            #
            u_next = (
                t_k * target_param
                + (1.0 - t_k) * w_k
            )

            p.data.copy_(u_next)

            # EMA target update
            #
            # g(u_k)
            #
            target_param.mul_(1.0 - self.target_tau)
            target_param.add_(
                p.data,
                alpha=self.target_tau,
            )

            # commit Adam moments
            state["exp_avg"].copy_(state["new_m"])
            state["exp_avg_sq"].copy_(state["new_v"])

            # cleanup
            del state["u_k"]
            del state["v_k"]
            del state["F_u_k"]
            del state["new_m"]
            del state["new_v"]

        self.current_lr = lambda_k

        self.iteration += 1

        self.last_stats = {
            "lambda_k": lambda_k,
            "beta_k": beta_k,
            "t_k": t_k,
        }

    # ========================================================
    # full step
    # ========================================================

    @torch.no_grad()
    def step(self, closure=None):

        raise RuntimeError(
            "Use:\n"
            "1. backward()\n"
            "2. extrapolation_step()\n"
            "3. second forward/backward\n"
            "4. pc_update()\n"
        )
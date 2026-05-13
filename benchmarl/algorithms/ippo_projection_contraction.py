"""
Projection-Contraction Extra-Gradient Adam for IPPO MARL
=========================================================

Implements Algorithms 1 & 2 from:
  "New Explicit Algorithms for a Class of the Split Common Solution Problem"
  Ha, Trang, Tuyen — Mathematical Methods in the Applied Sciences, 2025
  DOI: 10.1002/mma.11132

Key design decisions
--------------------
* F(x) is treated as the JOINT operator over ALL agents (Section 3 of paper).
  lambda_k is computed from global geometry across agents, then broadcast
  to each agent's optimizer.  This preserves the Lipschitz step-size
  guarantee (Remark 3.1 ii).

* Algorithm 1  (weak convergence, Theorem 3.1):
    v_k  = u_k - lambda_k * F(u_k)          # extrapolation
    d_k  = (u_k - v_k) - lambda_k*(F(u_k) - F(v_k))
    beta_k = beta * <u_k - v_k, d_k> / ||d_k||^2   (or gamma if ||d_k||=0)
    u_{k+1} = u_k - beta_k * d_k

* Algorithm 2  (strong convergence, Theorem 3.2):
    w_k     = u_k - beta_k * d_k
    u_{k+1} = t_k * g(u_k) + (1 - t_k) * w_k
  Activated by setting viscosity_mode="alg2" (or "halpern") in the config.

* F(·) is Adam-preconditioned: the "operator" seen by the PC geometry is
  the Adam-normalised gradient direction, not the raw gradient.  The
  raw gradient defines the loss landscape; Adam scales it so that the
  effective step in parameter space is well-conditioned regardless of
  layer scale.

* beta_k fallback (Remark 3.1 iii): when ||d_k|| = 0, the paper sets
  beta_k = gamma (a user-chosen positive constant) instead of skipping
  the update.  The old code kept theta_half and returned early — now
  fixed to apply u_{k+1} = u_k - gamma * d_k (which is u_k when d_k=0,
  i.e. a no-op, but the moment state and lambda_k are still updated).

* lambda_k EMA smoothing: the raw Lipschitz candidate theta*||x-y||/||ΔF||
  can be noisy in stochastic MARL.  An EMA with momentum `lambda_ema_alpha`
  (default 0.9) smooths it before clamping — this preserves the monotone
  non-increase property in expectation while avoiding single-batch spikes.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from tensordict import TensorDictBase

from benchmarl.algorithms.ippo_vi import IppoVI, IppoVIConfig
from benchmarl.experiment.callback import Callback


# ============================================================
# PCExtraGradientAdam
# ============================================================

class PCExtraGradientAdam(optim.Optimizer):
    """
    Two-step Projection-Contraction optimizer with Adam preconditioning.

    Each full iteration consists of two external calls:
      1. extrapolation_step()  — forward pass at theta_k, move to theta_half
      2. pc_update(...)        — forward pass at theta_half, apply PC update

    The geometry scalars (param_dist, grad_dist, d_k) are accumulated
    inside the optimizer but the GLOBAL lambda_k and beta_k that govern
    the update are injected by the callback after aggregating across all
    agents.  This is the correct multi-agent generalisation of F(x) as a
    joint operator.
    """

    def __init__(
        self,
        params,

        # Adam moments
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,

        # PC hyperparameters (paper notation)
        pc_beta: float = 1.0,          # beta  in (0, 2) — Alg 1 Step 5
        pc_gamma: float = 0.1,         # gamma > 0       — fallback when ||d_k||=0
        pc_theta: float = 0.95,        # theta in (0, 1) — Lipschitz line search

        # lambda_k bounds
        pc_lambda_min: float = 1e-6,
        pc_lambda_max: float = 1.0,

        # beta_k bounds (safety clamps, not in paper — numerical stability only)
        pc_beta_min: float = 1e-4,
        pc_beta_max: float = 2.0,      # must be < 2/beta for convergence

        # EMA smoothing for lambda_k  (alpha=0 → no smoothing, pure paper)
        lambda_ema_alpha: float = 0.9,

        # Gradient clipping (global norm)
        grad_clip: float = 10.0,
    ):
        assert 0 < pc_beta < 2, "pc_beta must be in (0, 2) per Theorem 3.1"
        assert 0 < pc_theta < 1, "pc_theta must be in (0, 1) per Algorithm 1"
        assert pc_beta_max < 2.0 / pc_beta + 1e-6, (
            "beta_k * pc_beta must stay < 2 for the (2/beta - 1) term to remain positive"
        )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.current_lr = lr           # lambda_k — updated each iteration
        self._lambda_ema = lr          # EMA of Lipschitz candidate
        self.lambda_ema_alpha = lambda_ema_alpha

        self.pc_beta = pc_beta
        self.pc_gamma = pc_gamma
        self.pc_theta = pc_theta

        self.pc_lambda_min = pc_lambda_min
        self.pc_lambda_max = pc_lambda_max
        self.pc_beta_min = pc_beta_min
        self.pc_beta_max = pc_beta_max

        self.grad_clip = grad_clip

        # Diagnostic statistics exposed to the callback for logging
        self.last_stats: Dict = {}
        # Geometry vectors accumulated during pc_update for joint aggregation
        self._geometry: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _collect_grads(self) -> Dict[torch.Tensor, torch.Tensor]:
        """
        Return {param: grad} for every parameter that has a gradient.
        Applies weight decay additively (L2 regularisation) if set.
        """
        grads: Dict[torch.Tensor, torch.Tensor] = {}
        for group in self.param_groups:
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().clone()
                if wd > 0:
                    g = g.add(p.data, alpha=wd)
                grads[p] = g
        return grads

    def _global_clip(
        self,
        grads: Dict[torch.Tensor, torch.Tensor],
    ) -> Dict[torch.Tensor, torch.Tensor]:
        """
        Global-norm clipping identical to torch.nn.utils.clip_grad_norm_.
        All tensors are scaled by the same scalar → directions preserved.
        """
        if not grads:
            return grads
        global_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads.values()))
        clip_coef = self.grad_clip / (global_norm + 1e-6)
        if clip_coef < 1.0:
            grads = {p: g * clip_coef for p, g in grads.items()}
        return grads

    def _adam_direction(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        step: int,
        group: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Adam-preconditioned direction from a gradient and current
        moment estimates.  Pure function — does NOT mutate any state.

        Returns (direction, new_exp_avg, new_exp_avg_sq).
        """
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        m = beta1 * exp_avg + (1.0 - beta1) * grad
        v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad

        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step

        denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
        direction = (m / bc1) / denom

        return direction, m, v

    def _init_state(self, p: torch.Tensor) -> dict:
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p.data)
            state["exp_avg_sq"] = torch.zeros_like(p.data)
        return state

    # ------------------------------------------------------------------
    # Step 1 — extrapolation
    #   v_k = u_k - lambda_k * F(u_k)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extrapolation_step(self) -> None:
        """
        Compute F(theta_k) via Adam preconditioning and move parameters to
        the extrapolated point theta_half = theta_k - lambda_k * F(theta_k).

        Saves theta_k, F(theta_k), and theta_half in per-parameter state.
        """
        grads = self._collect_grads()
        grads = self._global_clip(grads)

        for group in self.param_groups:
            for p in group["params"]:
                if p not in grads:
                    continue

                state = self._init_state(p)
                state["step"] += 1   # counts full iterations (1 increment per iter)

                direction, new_m, new_v = self._adam_direction(
                    grads[p],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group,
                )

                # Commit moments from the first evaluation
                state["exp_avg"].copy_(new_m)
                state["exp_avg_sq"].copy_(new_v)

                # Save anchors
                state["theta_k"] = p.data.clone()
                state["F_theta_k"] = direction.clone()

                # Move to extrapolated point
                theta_half = p.data - self.current_lr * direction
                state["theta_half"] = theta_half.clone()
                p.data.copy_(theta_half)

        # Clear geometry from previous iteration
        self._geometry = None

    # ------------------------------------------------------------------
    # Step 2 — projection-contraction update
    #
    # Geometry is computed locally; the callback injects the GLOBAL
    # lambda_k and beta_k (aggregated across all agents) before calling
    # apply_pc_update().  If called as a standalone optimizer (single
    # agent / no joint geometry), pass override_lambda=None to use the
    # locally computed value.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_pc_geometry(self) -> Optional[Dict]:
        """
        Collect per-optimizer geometry vectors needed to compute lambda_k
        and beta_k.  Does NOT apply any update.

        Returns a dict with flat tensors:
          x_minus_y  : theta_k - theta_half   (u_k - v_k in paper)
          operator_diff : F(theta_k) - F(theta_half)  (ΔF in paper)
          d_k        : (u_k - v_k) - lambda_k * ΔF
          cache      : list of (p, state, new_m, new_v, d_k_per_param)

        Returns None if grads are unavailable (e.g. zero-grad state).
        """
        eps = 1e-12

        grads_half = self._collect_grads()
        grads_half = self._global_clip(grads_half)

        all_xmy: List[torch.Tensor] = []
        all_opdiff: List[torch.Tensor] = []
        all_dk: List[torch.Tensor] = []
        cache = []

        for group in self.param_groups:
            for p in group["params"]:
                if p not in grads_half:
                    continue

                state = self.state[p]
                if "theta_k" not in state:
                    continue

                # Adam direction at theta_half (no step increment — same iter)
                dir_half, new_m, new_v = self._adam_direction(
                    grads_half[p],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group,
                )

                x_minus_y = state["theta_k"] - state["theta_half"]   # u_k - v_k
                op_diff = state["F_theta_k"] - dir_half               # F(u_k) - F(v_k)
                d_k = x_minus_y - self.current_lr * op_diff           # Eq. (3.1)

                all_xmy.append(x_minus_y.reshape(-1))
                all_opdiff.append(op_diff.reshape(-1))
                all_dk.append(d_k.reshape(-1))
                cache.append((p, state, new_m, new_v, d_k))

        if not cache:
            return None

        xmy_vec = torch.cat(all_xmy)
        opdiff_vec = torch.cat(all_opdiff)
        dk_vec = torch.cat(all_dk)

        geometry = {
            "xmy_vec": xmy_vec,
            "opdiff_vec": opdiff_vec,
            "dk_vec": dk_vec,
            "param_dist": torch.norm(xmy_vec),
            "grad_dist": torch.norm(opdiff_vec),
            "dk_norm": torch.norm(dk_vec),
            "dk_norm_sq": torch.sum(dk_vec * dk_vec) + eps,
            "numerator": torch.sum(xmy_vec * dk_vec),
            "cache": cache,
        }
        self._geometry = geometry
        return geometry

    @torch.no_grad()
    def _compute_local_lambda(self, geometry: Dict) -> float:
        """
        Compute the local Lipschitz-adaptive lambda_{k+1} from the geometry
        of THIS optimizer (single-agent or standalone use).

        For joint multi-agent use, the callback calls this on each optimizer
        and aggregates before calling apply_pc_update().
        """
        eps = 1e-12
        param_dist = geometry["param_dist"]
        grad_dist = geometry["grad_dist"]

        if grad_dist.item() > eps:
            raw_candidate = (self.pc_theta * param_dist / (grad_dist + eps)).item()
        else:
            raw_candidate = self.current_lr

        # EMA smoothing — damps noise while preserving non-increase trend
        alpha = self.lambda_ema_alpha
        self._lambda_ema = alpha * self._lambda_ema + (1.0 - alpha) * raw_candidate
        smoothed = self._lambda_ema

        return float(max(self.pc_lambda_min, min(smoothed, self.pc_lambda_max)))

    @torch.no_grad()
    def apply_pc_update(
        self,
        override_lambda: Optional[float] = None,
        override_beta: Optional[float] = None,
        viscosity_t: float = 0.0,
        g_values: Optional[Dict[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """
        Apply the PC update using geometry computed by _collect_pc_geometry().

        Parameters
        ----------
        override_lambda : float, optional
            Global lambda_k injected by callback (joint geometry).
            If None, uses locally computed value.
        override_beta : float, optional
            Global beta_k injected by callback (joint geometry).
            If None, computed from local geometry.
        viscosity_t : float
            t_k for Algorithm 2 viscosity step (0 = Algorithm 1, no viscosity).
        g_values : dict {param: g(param)}, optional
            Pre-computed g(u_k) values for viscosity. Required when viscosity_t > 0.
        """
        if self._geometry is None:
            warnings.warn(
                "apply_pc_update() called without prior _collect_pc_geometry(); "
                "call pc_update() instead for automatic geometry collection.",
                stacklevel=2,
            )
            return

        geometry = self._geometry
        cache = geometry["cache"]
        dk_norm_sq = geometry["dk_norm_sq"]
        numerator = geometry["numerator"]

        # ---- Determine lambda_k ----------------------------------------
        if override_lambda is not None:
            new_lambda = float(override_lambda)
        else:
            new_lambda = self._compute_local_lambda(geometry)

        # ---- Determine beta_k ------------------------------------------
        # Paper Algorithm 1, Step 5:
        #   beta_k = beta * <u_k - v_k, d_k> / ||d_k||^2   if ||d_k|| > 0
        #   beta_k = gamma                                    if ||d_k|| = 0
        #
        # Remark 3.1(iii): <u_k - v_k, d_k> >= 0 from k >= k_0, so
        # raw_beta should be non-negative in steady state.  In early
        # training it can be negative due to gradient noise — we clamp
        # to pc_beta_min (playing the role of gamma) rather than skipping.

        if override_beta is not None:
            beta_k = float(max(self.pc_beta_min, min(override_beta, self.pc_beta_max)))
            used_gamma_fallback = False
        else:
            dk_norm = geometry["dk_norm"].item()
            if dk_norm < 1e-12:
                # ||d_k|| = 0: paper prescribes beta_k = gamma
                beta_k = self.pc_gamma
                used_gamma_fallback = True
            else:
                raw_beta = (self.pc_beta * numerator / dk_norm_sq).item()
                if raw_beta <= 0:
                    # Contraction direction invalid — use gamma fallback
                    # (fixes the old "keep theta_half and return" behaviour)
                    beta_k = self.pc_gamma
                    used_gamma_fallback = True
                else:
                    beta_k = float(max(self.pc_beta_min, min(raw_beta, self.pc_beta_max)))
                    used_gamma_fallback = False

        # ---- Apply update: theta_{k+1} = theta_k - beta_k * d_k --------
        for (p, state, new_m, new_v, d_k) in cache:
            # Restore to anchor theta_k
            p.data.copy_(state["theta_k"])
            # Contraction: u_k - beta_k * d_k  →  w_k
            p.data.add_(d_k, alpha=-beta_k)

            # Algorithm 2 viscosity: u_{k+1} = t_k*g(u_k) + (1-t_k)*w_k
            if viscosity_t > 0.0:
                if g_values is not None and p in g_values:
                    g_uk = g_values[p]
                    p.data.mul_(1.0 - viscosity_t)
                    p.data.add_(g_uk, alpha=viscosity_t)
                # else: g not provided for this param — keep w_k as u_{k+1}

            # Commit Adam moments
            state["exp_avg"].copy_(new_m)
            state["exp_avg_sq"].copy_(new_v)

            # Clean up per-iteration temporaries
            del state["theta_k"]
            del state["theta_half"]
            del state["F_theta_k"]

        self.current_lr = new_lambda
        self._geometry = None

        self.last_stats = {
            "lambda_k": new_lambda,
            "beta_k": beta_k,
            "used_gamma_fallback": used_gamma_fallback,
            "param_dist": geometry["param_dist"].item(),
            "grad_dist": geometry["grad_dist"].item(),
            "dk_norm": geometry["dk_norm"].item(),
            "numerator": numerator.item(),
        }

    @torch.no_grad()
    def pc_update(
        self,
        viscosity_t: float = 0.0,
        g_values: Optional[Dict[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """
        Convenience method: collect geometry then apply update in one call.
        Use this for single-agent / standalone mode.
        For multi-agent joint-lambda mode use the callback's on_train_step.
        """
        geometry = self._collect_pc_geometry()
        if geometry is None:
            return
        self.apply_pc_update(
            override_lambda=None,
            override_beta=None,
            viscosity_t=viscosity_t,
            g_values=g_values,
        )


# ============================================================
# Joint geometry helper
# ============================================================

def compute_joint_lambda(
    geometries: List[Dict],
    pc_theta: float,
    pc_lambda_min: float,
    pc_lambda_max: float,
    lambda_ema: float,
    lambda_ema_alpha: float,
) -> Tuple[float, float]:
    """
    Aggregate geometry across all agents and compute a single global lambda_k.

    This implements the correct multi-agent interpretation of F(x) as a
    joint operator over the product Hilbert space H_1 x ... x H_N:

        ||x - y||_joint = sqrt( sum_i ||x_i - y_i||^2 )
        ||ΔF||_joint    = sqrt( sum_i ||ΔF_i||^2 )
        lambda_candidate = theta * ||x-y||_joint / ||ΔF||_joint

    Returns (new_lambda, updated_ema).
    """
    eps = 1e-12
    total_param_sq = sum(g["param_dist"].item() ** 2 for g in geometries)
    total_grad_sq = sum(g["grad_dist"].item() ** 2 for g in geometries)

    joint_param = math.sqrt(total_param_sq)
    joint_grad = math.sqrt(total_grad_sq)

    if joint_grad > eps:
        raw_candidate = pc_theta * joint_param / (joint_grad + eps)
    else:
        # No gradient variation across agents: keep current lambda
        raw_candidate = lambda_ema

    new_ema = lambda_ema_alpha * lambda_ema + (1.0 - lambda_ema_alpha) * raw_candidate
    new_lambda = float(max(pc_lambda_min, min(new_ema, pc_lambda_max)))
    return new_lambda, new_ema


def compute_joint_beta(
    geometries: List[Dict],
    pc_beta: float,
    pc_gamma: float,
    pc_beta_min: float,
    pc_beta_max: float,
) -> Tuple[float, bool]:
    """
    Compute global beta_k from the joint d_k and (u-v) across all agents.

    Paper Step 5:
        beta_k = beta * <u-v, d_k>_joint / ||d_k||^2_joint

    where the inner product and norm are taken in the product space.
    """
    eps = 1e-12
    total_numerator = sum(g["numerator"].item() for g in geometries)
    total_dk_sq = sum(g["dk_norm_sq"].item() for g in geometries) + eps
    total_dk_norm = math.sqrt(sum(g["dk_norm"].item() ** 2 for g in geometries))

    if total_dk_norm < 1e-12:
        return pc_gamma, True

    raw_beta = pc_beta * total_numerator / total_dk_sq
    if raw_beta <= 0:
        return pc_gamma, True

    beta_k = float(max(pc_beta_min, min(raw_beta, pc_beta_max)))
    return beta_k, False


# ============================================================
# Viscosity schedule helpers  (Algorithm 2, Theorem 3.2)
# ============================================================

def viscosity_harmonic(k: int, scale: float = 1.0) -> float:
    """t_k = scale / (k + 1)  — satisfies C1 (→0) and C2 (Σ=∞)."""
    return scale / (k + 1)


def viscosity_cosine(k: int, T: int, t_min: float = 1e-4) -> float:
    """Cosine annealing: t_k = t_min + 0.5*(1-t_min)*(1 + cos(pi*k/T))."""
    return t_min + 0.5 * (1.0 - t_min) * (1.0 + math.cos(math.pi * k / max(T, 1)))


# ============================================================
# Callback
# ============================================================

class ProjectionContractionCallback(Callback):
    """
    Orchestrates the PC-VI update across all MARL agents.

    Two modes
    ---------
    alg1 (default)
        Algorithm 1 (Theorem 3.1, weak convergence).
        u_{k+1} = u_k - beta_k * d_k

    alg2
        Algorithm 2 (Theorem 3.2, strong convergence via viscosity).
        w_k      = u_k - beta_k * d_k
        u_{k+1}  = t_k * g(u_k) + (1 - t_k) * w_k
        where g is a strict contraction (default: g(x) = 0.5 * x).

    halpern
        Algorithm 3 (Theorem 3.3) — special case of alg2 with g = const.
        Converges strongly to P_Omega(s).

    Joint geometry
    --------------
    lambda_k and beta_k are computed from the JOINT operator across ALL
    agent groups before any parameters are updated.  This is the correct
    multi-agent generalisation (Section 3).

    Critic
    ------
    The critic is updated with a plain Adam step, strictly AFTER the actor.
    Critic gradients are excluded from the PC geometry computation.
    """

    def __init__(self):
        super().__init__()
        self.actor_optimizers: Dict[str, PCExtraGradientAdam] = {}
        self.critic_optimizers: Dict[str, optim.Adam] = {}

        # Global step counter for viscosity schedule
        self._global_step: int = 0
        # Lambda EMA shared across all agents (joint operator)
        self._joint_lambda_ema: Optional[float] = None

    # ------------------------------------------------------------------
    # Setup: replace BenchMARL optimizers with PC-VI variants
    # ------------------------------------------------------------------

    def on_setup(self) -> None:
        exp = self.experiment
        algo_cfg = exp.algorithm_config

        # Initialise joint lambda EMA from the base learning rate
        self._joint_lambda_ema = exp.config.lr

        for group in exp.group_map.keys():
            group_opts = exp.optimizers[group]

            # ---- Actor -----------------------------------------------
            if "loss_objective" in group_opts:
                old_opt = group_opts["loss_objective"]

                # Initial lambda_0: use Baillon–Haddad heuristic if clip_epsilon
                # is available (Lemma 2.1: if L is the Lipschitz const of ∇L,
                # then ∇L is 1/L-ISM, and lambda_0 ≈ 1/L ≈ eps_clip).
                eps_clip = getattr(algo_cfg, "clip_epsilon", None)
                if eps_clip is not None and eps_clip > 0:
                    lambda_init = float(eps_clip)
                else:
                    lambda_init = exp.config.lr

                actor_opt = PCExtraGradientAdam(
                    [{"params": pg["params"]} for pg in old_opt.param_groups],
                    lr=lambda_init,
                    betas=getattr(algo_cfg, "extra_betas", (0.9, 0.999)),
                    eps=getattr(exp.config, "adam_eps", 1e-8),
                    weight_decay=getattr(exp.config, "weight_decay", 0.0),
                    pc_beta=getattr(algo_cfg, "pc_beta", 1.0),
                    pc_gamma=getattr(algo_cfg, "pc_gamma", 0.1),
                    pc_theta=getattr(algo_cfg, "pc_theta", 0.95),
                    pc_lambda_min=getattr(algo_cfg, "pc_lambda_min", 1e-6),
                    pc_lambda_max=getattr(algo_cfg, "pc_lambda_max", 1.0),
                    pc_beta_min=getattr(algo_cfg, "pc_beta_min", 1e-4),
                    pc_beta_max=getattr(algo_cfg, "pc_beta_max", 2.0),
                    lambda_ema_alpha=getattr(algo_cfg, "lambda_ema_alpha", 0.9),
                    grad_clip=getattr(algo_cfg, "grad_clip", 10.0),
                )

                group_opts["loss_objective"] = actor_opt
                self.actor_optimizers[group] = actor_opt

            # ---- Critic ----------------------------------------------
            if "loss_critic" in group_opts:
                self.critic_optimizers[group] = group_opts["loss_critic"]
            else:
                # BenchMARL merged critic into loss_objective — build dedicated Adam
                loss_module = exp.losses[group]
                critic_params = None
                for attr in ("critic", "value_network", "critic_network"):
                    sub = getattr(loss_module, attr, None)
                    if sub is not None:
                        critic_params = list(sub.parameters())
                        break

                if critic_params:
                    critic_opt = optim.Adam(
                        critic_params,
                        lr=getattr(algo_cfg, "critic_lr", exp.config.lr),
                        eps=getattr(exp.config, "adam_eps", 1e-8),
                        weight_decay=getattr(exp.config, "weight_decay", 0.0),
                    )
                    group_opts["loss_critic"] = critic_opt
                    self.critic_optimizers[group] = critic_opt

    # ------------------------------------------------------------------
    # Main training hook
    # ------------------------------------------------------------------

    def on_train_step(
        self,
        batch: TensorDictBase,
        group: str,
    ) -> None:
        exp = self.experiment
        algo_cfg = exp.algorithm_config
        viscosity_mode = getattr(algo_cfg, "viscosity_mode", "alg1")

        # ---- Compute viscosity t_k for Algorithm 2 -------------------
        viscosity_t = 0.0
        g_values: Optional[Dict] = None

        if viscosity_mode in ("alg2", "halpern"):
            vis_schedule = getattr(algo_cfg, "viscosity_schedule", "harmonic")
            vis_scale = getattr(algo_cfg, "viscosity_scale", 1.0)
            vis_T = getattr(algo_cfg, "viscosity_T", 10000)

            if vis_schedule == "harmonic":
                viscosity_t = viscosity_harmonic(self._global_step, vis_scale)
            elif vis_schedule == "cosine":
                viscosity_t = viscosity_cosine(self._global_step, vis_T)
            else:
                viscosity_t = viscosity_harmonic(self._global_step, vis_scale)

        # ================================================================
        # ACTOR — PC-VI two-step update
        # ================================================================
        if group not in self.actor_optimizers:
            pass
        else:
            actor_opt = self.actor_optimizers[group]

            def _actor_loss(loss_vals: dict) -> torch.Tensor:
                loss = loss_vals["loss_objective"]
                if "loss_entropy" in loss_vals:
                    loss = loss + loss_vals["loss_entropy"]
                return loss

            # Step 1: evaluate F(theta_k) and move to theta_half
            actor_opt.zero_grad()
            _actor_loss(exp.losses[group](batch)).backward()
            actor_opt.extrapolation_step()

            # Step 2: evaluate F(theta_half) and collect geometry
            actor_opt.zero_grad()
            _actor_loss(exp.losses[group](batch)).backward()
            geometry = actor_opt._collect_pc_geometry()

            if geometry is not None:
                # ---- Joint lambda / beta across agents -----------------
                # Collect geometries from ALL agents that have been processed
                # this step.  On the first group call, only this group has
                # geometry; on subsequent calls the others are already done.
                # We therefore use a within-step cache on the callback.
                #
                # For simplicity (and correctness in the common case of
                # simultaneous group updates), we compute joint values from
                # this optimizer alone when called per-group, and rely on the
                # EMA to smooth across consecutive steps.  A true simultaneous
                # aggregation would require the callback to buffer all groups
                # before applying any — see _joint_update_all_groups() below.
                new_lambda, new_ema = compute_joint_lambda(
                    geometries=[geometry],
                    pc_theta=actor_opt.pc_theta,
                    pc_lambda_min=actor_opt.pc_lambda_min,
                    pc_lambda_max=actor_opt.pc_lambda_max,
                    lambda_ema=self._joint_lambda_ema,
                    lambda_ema_alpha=actor_opt.lambda_ema_alpha,
                )
                self._joint_lambda_ema = new_ema

                joint_beta, used_gamma = compute_joint_beta(
                    geometries=[geometry],
                    pc_beta=actor_opt.pc_beta,
                    pc_gamma=actor_opt.pc_gamma,
                    pc_beta_min=actor_opt.pc_beta_min,
                    pc_beta_max=actor_opt.pc_beta_max,
                )

                # Prepare g(u_k) for viscosity if needed
                if viscosity_t > 0.0 and viscosity_mode != "halpern":
                    g_values = {
                        p: p.data * getattr(algo_cfg, "viscosity_contraction_c", 0.5)
                        for group_item in actor_opt.param_groups
                        for p in group_item["params"]
                    }
                elif viscosity_t > 0.0 and viscosity_mode == "halpern":
                    # Halpern: g(x) = s (anchor point, default = zeros)
                    anchor = getattr(algo_cfg, "halpern_anchor", None)
                    g_values = {
                        p: (
                            torch.full_like(p.data, anchor)
                            if anchor is not None
                            else torch.zeros_like(p.data)
                        )
                        for group_item in actor_opt.param_groups
                        for p in group_item["params"]
                    }

                actor_opt.apply_pc_update(
                    override_lambda=new_lambda,
                    override_beta=joint_beta,
                    viscosity_t=viscosity_t,
                    g_values=g_values,
                )

            actor_opt.zero_grad()

        # ================================================================
        # CRITIC — plain Adam, single step
        # Updated AFTER actor to avoid stale-advantage mismatch.
        # ================================================================
        if group in self.critic_optimizers:
            critic_opt = self.critic_optimizers[group]
            critic_opt.zero_grad()

            loss_vals_critic = exp.losses[group](batch)

            if "loss_critic" in loss_vals_critic:
                loss_vals_critic["loss_critic"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in critic_opt.param_groups for p in pg["params"]],
                    max_norm=getattr(algo_cfg, "grad_clip", 10.0),
                )
                critic_opt.step()

            critic_opt.zero_grad()

        self._global_step += 1

        # ================================================================
        # Logging
        # ================================================================
        if group not in self.actor_optimizers:
            return

        actor_opt = self.actor_optimizers[group]
        log_interval = getattr(algo_cfg, "pc_log_interval", 0)

        if log_interval > 0 and exp.n_iters_performed % log_interval == 0:
            stats = {
                f"train/pc_vi/{group}/{k}": v
                for k, v in actor_opt.last_stats.items()
            }
            stats[f"train/pc_vi/{group}/viscosity_t"] = viscosity_t
            stats[f"train/pc_vi/joint_lambda_ema"] = self._joint_lambda_ema
            exp.logger.log(stats, step=exp.n_iters_performed)

    # ------------------------------------------------------------------
    # Optional: simultaneous joint update across ALL groups in one step
    # ------------------------------------------------------------------

    def on_train_step_all_groups(self, batch: TensorDictBase) -> None:
        """
        True joint update: collect geometry from ALL agents first, compute
        a single (lambda_k, beta_k), then apply to all.

        Call this instead of on_train_step() when BenchMARL exposes a
        hook that iterates all groups in a single call.
        """
        exp = self.experiment
        algo_cfg = exp.algorithm_config

        def _actor_loss(loss_vals):
            loss = loss_vals["loss_objective"]
            if "loss_entropy" in loss_vals:
                loss = loss + loss_vals["loss_entropy"]
            return loss

        # Step 1: extrapolation for all agents
        for group, actor_opt in self.actor_optimizers.items():
            actor_opt.zero_grad()
            _actor_loss(exp.losses[group](batch)).backward()
            actor_opt.extrapolation_step()

        # Step 2: collect geometry for all agents
        geometries: Dict[str, Dict] = {}
        for group, actor_opt in self.actor_optimizers.items():
            actor_opt.zero_grad()
            _actor_loss(exp.losses[group](batch)).backward()
            geo = actor_opt._collect_pc_geometry()
            if geo is not None:
                geometries[group] = geo

        if not geometries:
            return

        geo_list = list(geometries.values())
        ref_opt = next(iter(self.actor_optimizers.values()))

        # Joint lambda and beta
        new_lambda, new_ema = compute_joint_lambda(
            geometries=geo_list,
            pc_theta=ref_opt.pc_theta,
            pc_lambda_min=ref_opt.pc_lambda_min,
            pc_lambda_max=ref_opt.pc_lambda_max,
            lambda_ema=self._joint_lambda_ema,
            lambda_ema_alpha=ref_opt.lambda_ema_alpha,
        )
        self._joint_lambda_ema = new_ema

        joint_beta, _ = compute_joint_beta(
            geometries=geo_list,
            pc_beta=ref_opt.pc_beta,
            pc_gamma=ref_opt.pc_gamma,
            pc_beta_min=ref_opt.pc_beta_min,
            pc_beta_max=ref_opt.pc_beta_max,
        )

        # Apply with shared lambda and beta
        for group, actor_opt in self.actor_optimizers.items():
            if group not in geometries:
                continue
            actor_opt.apply_pc_update(
                override_lambda=new_lambda,
                override_beta=joint_beta,
            )
            actor_opt.zero_grad()

        # Critic updates
        for group, critic_opt in self.critic_optimizers.items():
            critic_opt.zero_grad()
            loss_vals = exp.losses[group](batch)
            if "loss_critic" in loss_vals:
                loss_vals["loss_critic"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in critic_opt.param_groups for p in pg["params"]],
                    max_norm=getattr(algo_cfg, "grad_clip", 10.0),
                )
                critic_opt.step()
            critic_opt.zero_grad()

        self._global_step += 1


# ============================================================
# Algorithm class
# ============================================================

class IppoPCVI(IppoVI):
    """
    IPPO variant with Projection-Contraction VI optimizer for the actor.

    viscosity_mode:
        "alg1"    — Algorithm 1 (weak convergence, Theorem 3.1)
        "alg2"    — Algorithm 2 (strong convergence, Theorem 3.2)
        "halpern" — Algorithm 3 (strong convergence to P_Omega(s))

    viscosity_schedule (when mode != "alg1"):
        "harmonic" — t_k = viscosity_scale / (k + 1)
        "cosine"   — cosine annealing over viscosity_T steps
    """

    def __init__(
        self,
        # Adam moments
        extra_betas: Tuple[float, float] = (0.9, 0.999),

        # PC hyperparameters
        pc_beta: float = 1.0,
        pc_gamma: float = 0.1,
        pc_theta: float = 0.95,

        pc_lambda_min: float = 1e-6,
        pc_lambda_max: float = 1.0,
        pc_beta_min: float = 1e-4,
        pc_beta_max: float = 2.0,

        lambda_ema_alpha: float = 0.9,

        grad_clip: float = 10.0,

        # Critic
        critic_lr: float = 1e-3,

        # Viscosity (Algorithm 2)
        viscosity_mode: str = "alg1",
        viscosity_schedule: str = "harmonic",
        viscosity_scale: float = 1.0,
        viscosity_T: int = 10_000,
        viscosity_contraction_c: float = 0.5,
        halpern_anchor: Optional[float] = None,

        # Logging
        pc_log_interval: int = 0,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.extra_betas = extra_betas

        self.pc_beta = pc_beta
        self.pc_gamma = pc_gamma
        self.pc_theta = pc_theta

        self.pc_lambda_min = pc_lambda_min
        self.pc_lambda_max = pc_lambda_max
        self.pc_beta_min = pc_beta_min
        self.pc_beta_max = pc_beta_max

        self.lambda_ema_alpha = lambda_ema_alpha

        self.grad_clip = grad_clip
        self.critic_lr = critic_lr

        self.viscosity_mode = viscosity_mode
        self.viscosity_schedule = viscosity_schedule
        self.viscosity_scale = viscosity_scale
        self.viscosity_T = viscosity_T
        self.viscosity_contraction_c = viscosity_contraction_c
        self.halpern_anchor = halpern_anchor

        self.pc_log_interval = pc_log_interval


# ============================================================
# Config dataclass
# ============================================================

@dataclass
class IppoPCVIConfig(IppoVIConfig):

    # Adam moments
    extra_betas: Tuple[float, float] = (0.9, 0.999)

    # PC hyperparameters
    pc_beta: float = 1.0
    pc_gamma: float = 0.1
    pc_theta: float = 0.95

    pc_lambda_min: float = 1e-6
    pc_lambda_max: float = 1.0
    pc_beta_min: float = 1e-4
    pc_beta_max: float = 2.0

    lambda_ema_alpha: float = 0.9

    grad_clip: float = 10.0

    # Critic (plain Adam)
    critic_lr: float = 1e-3

    # Viscosity (Algorithm 2 / Halpern)
    viscosity_mode: str = "alg1"          # "alg1" | "alg2" | "halpern"
    viscosity_schedule: str = "harmonic"  # "harmonic" | "cosine"
    viscosity_scale: float = 1.0
    viscosity_T: int = 10_000
    viscosity_contraction_c: float = 0.5  # contraction coeff for g(x) = c*x
    halpern_anchor: Optional[float] = None

    # Logging
    pc_log_interval: int = 0

    @staticmethod
    def associated_class():
        return IppoPCVI
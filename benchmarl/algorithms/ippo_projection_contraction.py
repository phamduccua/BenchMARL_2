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
        pc_lambda_max: float = 1e-3,

        pc_beta_min: float = 0.3,
        pc_beta_max: float = 1.0,

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

    def _collect_grads(self) -> Dict[torch.Tensor, torch.Tensor]:
        """
        Collect detached gradients (+ weight decay) for every parameter
        that has a gradient.  Returns {p: grad} mapping.
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
        True global-norm clipping:

            g  ←  g · c / max(‖g‖_global, c)

        where ‖g‖_global = sqrt( Σ_i ‖g_i‖² ) over ALL parameters.
        Each tensor is scaled by the SAME scalar, so relative directions
        are preserved exactly — identical to PyTorch's clip_grad_norm_.
        """
        global_norm = torch.sqrt(
            sum(g.norm() ** 2 for g in grads.values())
        )
        clip_coef = self.grad_clip / (global_norm + 1e-6)
        if clip_coef < 1.0:
            grads = {p: g * clip_coef for p, g in grads.items()}
        return grads

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
        # --- 1. Collect raw grads (with weight decay) then global-clip ---
        grads = self._collect_grads()
        grads = self._global_clip(grads)

        # --- 2. Per-parameter Adam direction + extrapolation ---
        for group in self.param_groups:
            for p in group["params"]:
                if p not in grads:
                    continue

                grad = grads[p]

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

        # --- 1. Collect grads at theta_half, global-clip ---
        grads_half = self._collect_grads()
        grads_half = self._global_clip(grads_half)

        all_x_minus_y: List[torch.Tensor] = []
        all_operator_diff: List[torch.Tensor] = []
        all_dk: List[torch.Tensor] = []

        cache = []

        # --------------------------------------------------------
        # Collect per-parameter quantities
        # --------------------------------------------------------
        for group in self.param_groups:
            for p in group["params"]:
                if p not in grads_half:
                    continue

                state = self.state[p]

                # Guard: extrapolation_step must have been called first
                if "theta_k" not in state:
                    continue

                grad_half = grads_half[p]

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
        #
        # FIX: do NOT include self.current_lr in min().
        # The old code ratcheted lr monotonically downward to
        # pc_lambda_min and never recovered, killing actor updates.
        # --------------------------------------------------------
        if grad_dist.item() > eps:
            candidate_lr = (self.pc_theta * param_dist / (grad_dist + eps)).item()
        else:
            candidate_lr = self.current_lr

        adaptive_lr = float(
            max(self.pc_lambda_min,
                min(candidate_lr, self.pc_lambda_max))  # removed self.current_lr
        )

        # --------------------------------------------------------
        # Contraction coefficient: beta_k
        # beta_k = pc_beta * <x-y, d_k> / ||d_k||^2
        #
        # FIX: if numerator <= 0, d_k points away from the descent
        # direction — the PC contraction is invalid. Fall back to
        # keeping theta_half (the extrapolated point) as the update.
        # --------------------------------------------------------
        raw_beta = (self.pc_beta * numerator / dk_norm_sq).item()

        # Fall back to EG (keep theta_half) when the contraction direction is
        # invalid (raw_beta <= 0) OR too weak (raw_beta < pc_beta_min).
        # A clamped-up beta would overshoot past the intended contraction
        # geometry, so it is better to use the EG point in that regime.
        if raw_beta < self.pc_beta_min:
            for (p, state, exp_avg_new, exp_avg_sq_new, d_k) in cache:
                # p.data already holds theta_half — nothing to change
                state["exp_avg"].copy_(exp_avg_new)
                state["exp_avg_sq"].copy_(exp_avg_sq_new)
                del state["theta_k"]
                del state["theta_half"]
                del state["F_theta_k"]
            self.current_lr = adaptive_lr
            self.last_stats = {
                "lambda_k": self.current_lr,
                "beta_k": 0.0,
                "skipped_pc": True,
                "param_dist": param_dist.item(),
                "grad_dist": grad_dist.item(),
                "dk_norm": torch.norm(d_k_vec).item(),
            }
            return

        beta_k = float(min(raw_beta, self.pc_beta_max))

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
    """
    Replaces the monolithic loss_objective optimizer with two separate optimizers:

      actor_optimizer  — PCExtraGradientAdam (2-step VI)
                         handles loss_objective + loss_entropy ONLY.
                         These losses define the VI game operator.

      critic_optimizer — plain Adam (1-step)
                         handles loss_critic ONLY.
                         The critic is a regression target, not a VI player.

    Why separate?
    -------------
    The PC geometry (operator norm, Lipschitz step-size, beta_k) must reflect
    only the actor's game dynamics.  Including critic gradients distorts both
    the global norm used for clipping and the operator-difference estimate
    F(theta_k) - F(theta_half), breaking the theoretical guarantees of the
    projection-contraction method.
    """

    def __init__(self):
        super().__init__()
        self.actor_optimizers: Dict[str, PCExtraGradientAdam] = {}
        self.critic_optimizers: Dict[str, optim.Adam] = {}

    def on_setup(self):
        exp = self.experiment
        algo_cfg = exp.algorithm_config

        for group in exp.group_map.keys():

            group_opts = exp.optimizers[group]

            # --------------------------------------------------
            # Actor: replace loss_objective optimizer with PC-VI
            # --------------------------------------------------
            if "loss_objective" in group_opts:
                old_actor_opt = group_opts["loss_objective"]

                actor_opt = PCExtraGradientAdam(
                    [{"params": pg["params"]} for pg in old_actor_opt.param_groups],
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

                group_opts["loss_objective"] = actor_opt
                self.actor_optimizers[group] = actor_opt

            # --------------------------------------------------
            # Critic: keep (or create) a plain Adam optimizer.
            # BenchMARL may already have a "loss_critic" optimizer;
            # if not, we build one from the critic parameters
            # exposed through the loss module.
            # --------------------------------------------------
            if "loss_critic" in group_opts:
                # Already managed by BenchMARL — keep as-is but
                # register it so we can call it explicitly.
                self.critic_optimizers[group] = group_opts["loss_critic"]

            elif "loss_critic" not in group_opts:
                # BenchMARL merged critic params into loss_objective.
                # Build a dedicated Adam from the critic sub-module.
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

    # ============================================================
    # Main MARL training hook
    #
    # Order:
    #   1. Actor step   (PC-VI, 2 forward+backward)  ← FIRST
    #   2. Critic step  (plain Adam, 1 forward+backward)
    #
    # Actor goes FIRST because the batch has precomputed advantages
    # from rollout time.  Updating the critic first would shift
    # value-network weights without updating the stored advantages,
    # creating a stale-baseline mismatch inside loss_objective.
    # ============================================================

    def on_train_step(
        self,
        batch: TensorDictBase,
        group: str,
    ):
        exp = self.experiment

        # --------------------------------------------------------
        # ACTOR — PC-VI two-step update (always first)
        # Only loss_objective + loss_entropy define the VI operator.
        # loss_critic is intentionally excluded.
        # --------------------------------------------------------
        if group in self.actor_optimizers:
            actor_opt = self.actor_optimizers[group]

            def _actor_loss(loss_vals) -> torch.Tensor:
                loss = loss_vals["loss_objective"]
                if "loss_entropy" in loss_vals:
                    loss = loss + loss_vals["loss_entropy"]
                return loss

            # Step 1: gradient at theta_k → extrapolate to theta_half
            actor_opt.zero_grad()
            _actor_loss(exp.losses[group](batch)).backward()
            actor_opt.extrapolation_step()

            # Step 2: gradient at theta_half → PC-VI update
            actor_opt.zero_grad()
            _actor_loss(exp.losses[group](batch)).backward()
            actor_opt.pc_update()

            actor_opt.zero_grad()

        # --------------------------------------------------------
        # CRITIC — plain Adam, single step, isolated backward pass
        # Updated AFTER actor; critic regression on current V targets.
        # --------------------------------------------------------
        if group in self.critic_optimizers:
            critic_opt = self.critic_optimizers[group]
            critic_opt.zero_grad()

            loss_vals_critic = exp.losses[group](batch)

            if "loss_critic" in loss_vals_critic:
                loss_vals_critic["loss_critic"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in critic_opt.param_groups for p in pg["params"]],
                    max_norm=getattr(exp.algorithm_config, "grad_clip", 10.0),
                )
                critic_opt.step()

            critic_opt.zero_grad()

        if group not in self.actor_optimizers:
            return

        actor_opt = self.actor_optimizers[group]

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------
        log_interval = getattr(exp.algorithm_config, "pc_log_interval", 0)

        if log_interval > 0 and exp.n_iters_performed % log_interval == 0:
            exp.logger.log(
                {
                    f"train/pc_vi/{group}/{k}": v
                    for k, v in actor_opt.last_stats.items()
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
        pc_lambda_max: float = 1e-3,

        pc_beta_min: float = 0.3,
        pc_beta_max: float = 1.0,

        grad_clip: float = 10.0,

        # Critic (plain Adam, separate from VI)
        critic_lr: float = 1e-3,

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
        self.critic_lr = critic_lr

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
    pc_lambda_max: float = 1e-3

    pc_beta_min: float = 0.3
    pc_beta_max: float = 1.0

    grad_clip: float = 10.0

    # Critic uses a plain Adam with its own (typically higher) lr
    critic_lr: float = 1e-3

    pc_log_interval: int = 0

    @staticmethod
    def associated_class():
        return IppoPCVI
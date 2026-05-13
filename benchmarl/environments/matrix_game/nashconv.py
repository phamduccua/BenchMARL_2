#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Exact NashConv callback cho two-player zero-sum matrix games.

Tại sao không dùng NashConvCallback chung?
─────────────────────────────────────────
NashConvCallback chung dùng REINFORCE để train best-response (BR).
Với matrix game chỉ có 1 bước/episode, REINFORCE bị 2 vấn đề:

  1. Return normalisation triệt tiêu gradient:
     Nếu tất cả reward trong batch giống nhau (ví dụ BR đã hội tụ
     chơi Paper, mọi episode đều thắng +1), std = 0 → gradient = 0
     → BR không cập nhật được.

  2. Ước lượng gradient cực kỳ nhiễu: chỉ 1 reward/episode →
     phương sai cao → BR không hội tụ đáng tin cậy.

Giải pháp: tính NashConv CHÍNH XÁC từ policy logits + lý thuyết game.
──────────────────────────────────────────────────────────────────────
Matrix game có observation HẰNG SỐ (= zeros). Vì vậy, thay vì ước lượng
π₀, π₁ từ tần suất action trong rollouts, ta có thể query TRỰC TIẾP policy
network với obs=0 để lấy exact action probabilities (softmax của logits).

Cho game 2 agent với payoff matrix P (agent 0 tối đa hoá, agent 1 tối thiểu):

  π₀, π₁ = softmax(policy_logits)    ← CHÍNH XÁC, không sampling

  Baseline:  V₀  = π₀ᵀ P π₁          (kỳ vọng reward với policy hiện tại)
  BR agent0: BR₀ = max_a (P π₁)[a]    (hành động tốt nhất chống lại π₁)
  Gap₀      = max(0, BR₀ − V₀)

  Baseline:  V₁  = −V₀               (zero-sum)
  BR agent1: BR₁ = max_a (−Pᵀ π₀)[a] (hành động tốt nhất chống lại π₀)
  Gap₁      = max(0, BR₁ − V₁)

  NashConv  = Gap₀ + Gap₁            (= 0 khi cả hai đều ở Nash eq)

Với off-policy algorithms (IQL, QMIX, …) không có logits, fallback về
frequency estimation từ rollouts (vẫn chính xác với đủ mẫu).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Nash Equilibrium đã biết chính xác cho từng game (mixed strategy)
# RPS:              cả hai chơi uniform (1/3, 1/3, 1/3)
# Matching Pennies: cả hai chơi uniform (1/2, 1/2)
_NASH_EQ: Dict[str, torch.Tensor] = {
    "rps":              torch.tensor([1/3, 1/3, 1/3]),
    "matching_pennies": torch.tensor([0.5, 0.5]),
}

from benchmarl.experiment.callback import Callback


class MatrixGameNashConvCallback(Callback):
    """Tính NashConv và Distance-to-Equilibrium cho 2-agent zero-sum matrix game.

    Ưu tiên lấy π từ policy logits (chính xác tuyệt đối).
    Fallback về frequency estimation nếu algorithm không expose logits.

    Args:
        payoff:        Payoff matrix ``[n_actions, n_actions]`` cho agent 0.
        scenario:      ``"rps"`` hoặc ``"matching_pennies"`` — dùng để tra
                       Nash Equilibrium đã biết khi tính distance to eq.
        eval_interval: Tính sau mỗi bao nhiêu lần evaluation.
        min_samples:   Số mẫu tối thiểu cho fallback frequency estimation.
    """

    def __init__(
        self,
        payoff: torch.Tensor,
        scenario: str,
        eval_interval: int = 1,
        min_samples: int = 10,
    ):
        super().__init__()
        self._payoff_cpu = payoff.float().cpu()
        self.n_actions_0 = payoff.shape[0]
        self.n_actions_1 = payoff.shape[1]
        self.eval_interval = eval_interval
        self.min_samples = min_samples
        self._eval_count = 0

        # Nash Equilibrium đã biết — dùng để tính Distance to Equilibrium
        if scenario not in _NASH_EQ:
            raise ValueError(f"Scenario '{scenario}' chưa có Nash eq. Thêm vào _NASH_EQ.")
        self._nash_eq_cpu = _NASH_EQ[scenario].float().cpu()   # π* shape [n_actions]

    # ------------------------------------------------------------------
    # Callback hook
    # ------------------------------------------------------------------

    def on_evaluation_end(self, rollouts: List[TensorDictBase]) -> None:
        self._eval_count += 1
        if (self._eval_count - 1) % self.eval_interval != 0:
            return

        exp = self.experiment
        device = exp.config.train_device
        payoff = self._payoff_cpu.to(device)

        # ── Bước 1: Lấy π₀, π₁ ─────────────────────────────────────────
        # Ưu tiên: query trực tiếp policy logits (CHÍNH XÁC)
        # Fallback: ước lượng từ tần suất action trong rollouts
        pi_0, pi_1, source = self._get_action_probs(rollouts, device)
        if pi_0 is None:
            return  # Không đủ dữ liệu

        # ── Bước 2: Tính NashConv chính xác ─────────────────────────────
        # V0 = π₀ᵀ P π₁  (scalar)
        V0 = (pi_0 @ payoff @ pi_1)
        V1 = -V0                             # zero-sum

        # BR agent 0: max_a (P π₁)[a]
        br_vals_0 = payoff @ pi_1            # [n_actions_0]
        BR0 = br_vals_0.max()

        # BR agent 1: max_a (−Pᵀ π₀)[a]
        br_vals_1 = -(payoff.T @ pi_0)       # [n_actions_1]
        BR1 = br_vals_1.max()

        gap_0 = float(torch.clamp(BR0 - V0, min=0.0))
        gap_1 = float(torch.clamp(BR1 - V1, min=0.0))
        nashconv = gap_0 + gap_1

        # ── Bước 3: Distance to Equilibrium ─────────────────────────────
        # d = ||π₀ - π*||₂² + ||π₁ - π*||₂²
        pi_star = self._nash_eq_cpu.to(device)
        dist_0 = float(((pi_0 - pi_star) ** 2).sum())
        dist_1 = float(((pi_1 - pi_star) ** 2).sum())
        dist_total = dist_0 + dist_1

        # ── Bước 4: Log ──────────────────────────────────────────────────
        to_log: Dict[str, float] = {
            # NashConv
            "eval/nashconv/agents/gap_agent_0":       gap_0,
            "eval/nashconv/agents/gap_agent_1":       gap_1,
            "eval/nashconv/agents/nashconv":          nashconv,
            "eval/nashconv/agents/base_util_agent_0": float(V0),
            "eval/nashconv/agents/base_util_agent_1": float(V1),
            "eval/nashconv/agents/br_util_agent_0":   float(BR0),
            "eval/nashconv/agents/br_util_agent_1":   float(BR1),
            "eval/nashconv/total":                    nashconv,
            # Distance to Equilibrium
            "eval/distance_to_eq/agent_0":  dist_0,
            "eval/distance_to_eq/agent_1":  dist_1,
            "eval/distance_to_eq/total":    dist_total,
        }

        # Log xác suất hành động của từng agent
        for a in range(self.n_actions_0):
            to_log[f"eval/policy/agent_0_action_{a}"] = float(pi_0[a])
        for a in range(self.n_actions_1):
            to_log[f"eval/policy/agent_1_action_{a}"] = float(pi_1[a])

        exp.logger.log(to_log, step=exp.n_iters_performed)

        if exp.n_iters_performed % 10 == 0:
            print(f"[NashConv|{source}] iter={exp.n_iters_performed} "
                  f"nashconv={nashconv:.4f}  dist_eq={dist_total:.4f}  "
                  f"π₀={[f'{p:.3f}' for p in pi_0.tolist()]}  "
                  f"π₁={[f'{p:.3f}' for p in pi_1.tolist()]}")

    # ------------------------------------------------------------------
    # Lấy action probabilities: logits trước, fallback frequency
    # ------------------------------------------------------------------

    def _get_action_probs(
        self,
        rollouts: List[TensorDictBase],
        device: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """Trả về (pi_0, pi_1, source) trong đó source là 'logits' hoặc 'frequency'."""

        # ── Phương pháp 1: Policy logits (CHÍNH XÁC) ────────────────────
        pi_0, pi_1 = self._probs_from_logits(device)
        if pi_0 is not None:
            return pi_0, pi_1, "logits"

        # ── Phương pháp 2: Frequency estimation (fallback) ──────────────
        pi_0, pi_1 = self._probs_from_frequency(rollouts, device)
        if pi_0 is not None:
            return pi_0, pi_1, "frequency"

        return None, None, "none"

    def _probs_from_logits(
        self,
        device: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Lấy xác suất chính xác bằng cách forward policy với obs=zeros.

        Matrix game có observation hằng số = 0, nên chỉ cần 1 forward pass.
        Policy output logits tại key ("agents", "logits").
        softmax(logits[agent_0]) → π₀, softmax(logits[agent_1]) → π₁.

        Trả về (None, None) nếu algorithm không expose logits
        (ví dụ: off-policy Q-learning như IQL, QMIX).
        """
        try:
            exp = self.experiment

            # Tạo dummy TensorDict với obs = zeros
            # Shape: [B=1, n_agents=2, obs_dim=1]
            obs = torch.zeros(1, 2, 1, device=device)
            td = TensorDict(
                {
                    "agents": TensorDict(
                        {"observation": obs},
                        batch_size=[1, 2],
                    )
                },
                batch_size=[1],
            )

            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                out = exp.policy(td)

            # Lấy logits → softmax → [n_agents, n_actions]
            logits = out.get(("agents", "logits"), None)
            if logits is None:
                return None, None

            # logits shape: [B=1, n_agents=2, n_actions]
            probs = torch.softmax(logits[0], dim=-1)  # [n_agents, n_actions]
            pi_0 = probs[0].to(device)   # [n_actions_0]
            pi_1 = probs[1].to(device)   # [n_actions_1]
            return pi_0, pi_1

        except Exception as e:
            # Algorithm không hỗ trợ → fallback về frequency
            return None, None

    def _probs_from_frequency(
        self,
        rollouts: List[TensorDictBase],
        device: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Ước lượng π₀, π₁ từ tần suất hành động trong rollouts (fallback).

        Dùng cho off-policy algorithms (IQL, QMIX, MADDPG, …)
        không expose logits trực tiếp.
        """
        a0_list: List[torch.Tensor] = []
        a1_list: List[torch.Tensor] = []

        for td in rollouts:
            actions = td.get(("agents", "action"), None)
            if actions is None:
                continue
            # actions có thể có nhiều shape: [T, B, 2], [B, 2], [T, 2], …
            actions = actions.reshape(-1, actions.shape[-1]).long()
            a0_list.append(actions[:, 0])
            a1_list.append(actions[:, 1])

        if not a0_list:
            return None, None

        a0 = torch.cat(a0_list).to(device)
        a1 = torch.cat(a1_list).to(device)
        N = a0.shape[0]

        if N < self.min_samples:
            print(f"[MatrixGameNashConv] Bỏ qua frequency fallback: "
                  f"N={N} < min_samples={self.min_samples}")
            return None, None

        pi_0 = torch.zeros(self.n_actions_0, device=device)
        pi_1 = torch.zeros(self.n_actions_1, device=device)
        for a in range(self.n_actions_0):
            pi_0[a] = (a0 == a).float().sum()
        for a in range(self.n_actions_1):
            pi_1[a] = (a1 == a).float().sum()

        pi_0 = pi_0 / pi_0.sum()
        pi_1 = pi_1 / pi_1.sum()
        return pi_0, pi_1

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Two-player zero-sum matrix game environments (RPS, Matching Pennies)."""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    Categorical,
    Composite,
    Unbounded,
)
from torchrl.envs import EnvBase

# Payoff matrix for agent 0; agent 1 receives the negated reward (zero-sum).
# payoff[a0, a1] = reward for agent 0
_PAYOFFS = {
    # Rock=0, Paper=1, Scissors=2
    "rps": torch.tensor(
        [
            [0.0, -1.0,  1.0],   # Rock     beats Scissors, loses to Paper
            [1.0,  0.0, -1.0],   # Paper    beats Rock,     loses to Scissors
            [-1.0, 1.0,  0.0],   # Scissors beats Paper,   loses to Rock
        ]
    ),
    # Heads=0, Tails=1
    # Row player (agent 0) wins when actions MATCH
    "matching_pennies": torch.tensor(
        [
            [ 1.0, -1.0],
            [-1.0,  1.0],
        ]
    ),
}


class MatrixGameEnv(EnvBase):
    """Vectorised two-player zero-sum matrix game.

    Both agents choose simultaneously; the episode ends after ``max_steps``
    rounds (default 1 for a one-shot game).

    Observations are constant zeros — the game is stateless and symmetric.
    Actions are integer-encoded via ``Categorical`` spec.

    Nash equilibria (mixed strategies):
      - RPS:              (1/3, 1/3, 1/3) for both agents → NashConv = 0
      - Matching Pennies: (1/2, 1/2)     for both agents → NashConv = 0

    Args:
        scenario:  ``"rps"`` or ``"matching_pennies"``.
        num_envs:  number of parallel environments in the batch.
        max_steps: episode length (1 = one-shot game).
        seed:      optional RNG seed.
        device:    torch device string.
    """

    def __init__(
        self,
        scenario: str,
        num_envs: int = 1,
        max_steps: int = 1,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=[num_envs])
        if scenario not in _PAYOFFS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. Choose from {list(_PAYOFFS)}"
            )
        self.scenario = scenario
        self.max_steps = max_steps
        self._payoff = _PAYOFFS[scenario].to(device)
        self.n_actions = self._payoff.shape[0]
        # Per-env step counter for episode termination
        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._make_specs()
        if seed is not None:
            self.set_seed(seed)

    # ------------------------------------------------------------------
    # Spec construction  (shapes include leading batch dimensions)
    # ------------------------------------------------------------------

    def _make_specs(self) -> None:
        B = list(self.batch_size)  # [num_envs]
        n = self.n_actions

        # Observations: each agent receives a constant zero vector of size 1
        self.observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(shape=(*B, 2, 1), device=self.device),
                shape=(*B, 2),
            ),
            shape=B,
        )

        # Integer actions: each value in {0, …, n-1}
        self.action_spec = Composite(
            agents=Composite(
                action=Categorical(
                    n=n, shape=(*B, 2), dtype=torch.long, device=self.device
                ),
                shape=(*B, 2),
            ),
            shape=B,
        )

        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(shape=(*B, 2, 1), device=self.device),
                shape=(*B, 2),
            ),
            shape=B,
        )

        self.done_spec = Composite(
            done=Categorical(n=2, shape=(*B, 1), dtype=torch.bool, device=self.device),
            terminated=Categorical(
                n=2, shape=(*B, 1), dtype=torch.bool, device=self.device
            ),
            shape=B,
        )

    # ------------------------------------------------------------------
    # EnvBase interface
    # ------------------------------------------------------------------

    def _reset(self, tensordict=None) -> TensorDict:
        B = list(self.batch_size)          # [num_envs]
        self._step_count.zero_()
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": torch.zeros(
                            *B, 2, 1, dtype=torch.float32, device=self.device
                        )
                    },
                    batch_size=[*B, 2],
                ),
                "done": torch.zeros(*B, 1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(
                    *B, 1, dtype=torch.bool, device=self.device
                ),
            },
            batch_size=B,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        B = list(self.batch_size)

        # actions: [B, 2] integer
        actions = tensordict.get(("agents", "action"))
        a0 = actions[..., 0].long()  # [B]
        a1 = actions[..., 1].long()  # [B]

        r0 = self._payoff[a0, a1]    # [B]  reward for agent 0
        r1 = -r0                     # [B]  zero-sum

        rewards = torch.stack([r0, r1], dim=-1).unsqueeze(-1)  # [B, 2, 1]

        self._step_count += 1
        done = (self._step_count >= self.max_steps).unsqueeze(-1)  # [B, 1]

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": torch.zeros(
                            *B, 2, 1, dtype=torch.float32, device=self.device
                        ),
                        "reward": rewards,
                    },
                    batch_size=[*B, 2],
                ),
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=B,
        )

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(seed)

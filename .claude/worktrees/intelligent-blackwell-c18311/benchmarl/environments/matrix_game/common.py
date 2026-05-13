#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""BenchMARL task wrappers for matrix game environments."""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from .matrix_game_env import MatrixGameEnv


class MatrixGameClass(TaskClass):
    """TaskClass for RPS and Matching Pennies matrix games."""

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: MatrixGameEnv(
            scenario=self.name.lower(),
            num_envs=num_envs,
            seed=seed,
            device=device,
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"agents": ["agent_0", "agent_1"]}

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.full_observation_spec_unbatched

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "matrix_game"


class MatrixGameTask(Task):
    """Available matrix game tasks."""

    RPS = None
    MATCHING_PENNIES = None

    @staticmethod
    def associated_class():
        return MatrixGameClass

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from benchmarl.hydra_config import load_experiment_from_hydra

# ── Callback registry ────────────────────────────────────────────────────────
# Mỗi algorithm name → danh sách callback class cần bật tự động.
# Callbacks được khởi tạo mỗi lần chạy (không share state).

def _build_callbacks(algorithm_name: str, task_name: str, cfg: DictConfig):
    """Trả về danh sách callback instances phù hợp với algorithm và task."""
    callbacks = []

    # ── VIOptimizer callbacks ────────────────────────────────────────────────
    if algorithm_name == "ippo_vi":
        from benchmarl.algorithms.ippo_vi import VIOptimizerCallback
        callbacks.append(VIOptimizerCallback())

    elif algorithm_name == "ippo_extragradient":
        from benchmarl.algorithms.ippo_extragradient import ExtraGradientCallback
        callbacks.append(ExtraGradientCallback())

    elif algorithm_name == "ippo_vi_no_norm":
        from benchmarl.algorithms.ippo_vi_no_norm import VIOptimizerNoNormCallback
        callbacks.append(VIOptimizerNoNormCallback())

    elif algorithm_name == "ippo_vi_no_anchor":
        from benchmarl.algorithms.ippo_vi_no_anchor import VIOptimizerNoAnchorCallback
        callbacks.append(VIOptimizerNoAnchorCallback())

    # ── NashConv (opt-in qua config) ─────────────────────────────────────────
    nashconv_cfg = OmegaConf.select(cfg, "nashconv", default=None)
    if nashconv_cfg is not None and OmegaConf.select(nashconv_cfg, "enable", default=False):
        eval_interval = OmegaConf.select(nashconv_cfg, "eval_interval", default=1)

        if task_name.startswith("matrix_game/"):
            # ── Matrix game: tính NashConv CHÍNH XÁC từ action frequencies ──
            # REINFORCE không phù hợp với game 1 bước (gradient vanish).
            scenario = task_name.split("/")[1]   # "rps" hoặc "matching_pennies"
            from benchmarl.environments.matrix_game.matrix_game_env import _PAYOFFS
            from benchmarl.environments.matrix_game.nashconv import (
                MatrixGameNashConvCallback,
            )
            payoff = _PAYOFFS[scenario]
            callbacks.append(
                MatrixGameNashConvCallback(
                    payoff=payoff,
                    scenario=scenario,
                    eval_interval=eval_interval,
                    min_samples=OmegaConf.select(nashconv_cfg, "min_samples", default=50),
                )
            )
            print(f"[NashConv] MatrixGame exact mode  (scenario={scenario}, "
                  f"eval_interval={eval_interval})")
        else:
            # ── Env khác: dùng approximate NashConv (REINFORCE BR) ───────────
            from benchmarl.algorithms.nashconv_callback import NashConvCallback
            callbacks.append(
                NashConvCallback(
                    br_updates=OmegaConf.select(nashconv_cfg, "br_updates", default=5),
                    br_episodes=OmegaConf.select(nashconv_cfg, "br_episodes", default=4),
                    eval_episodes=OmegaConf.select(nashconv_cfg, "eval_episodes", default=5),
                    br_lr=OmegaConf.select(nashconv_cfg, "br_lr", default=3e-4),
                    br_hidden_dim=OmegaConf.select(nashconv_cfg, "br_hidden_dim", default=64),
                    entropy_coef=OmegaConf.select(nashconv_cfg, "entropy_coef", default=0.001),
                    gamma=OmegaConf.select(nashconv_cfg, "gamma", default=0.99),
                    eval_interval=eval_interval,
                    deterministic_eval=OmegaConf.select(
                        nashconv_cfg, "deterministic_eval", default=True
                    ),
                    obs_key=OmegaConf.select(nashconv_cfg, "obs_key", default="observation"),
                )
            )
            print(f"[NashConv] approximate mode  (br_updates={nashconv_cfg.get('br_updates', 5)}, "
                  f"eval_interval={eval_interval})")

    return callbacks


# ── Hydra main ───────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    """Runs an experiment loading its config from hydra.

    Basic usage::

        python benchmarl/run.py algorithm=mappo task=vmas/balance

    Enable NashConv::

        python benchmarl/run.py algorithm=ippo_vi task=vmas/balance \\
            nashconv.enable=true nashconv.br_updates=5 nashconv.eval_interval=2

    Args:
        cfg (DictConfig): the hydra config dictionary

    """
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    callbacks = _build_callbacks(algorithm_name, task_name, cfg)
    if callbacks:
        print(f"[Callbacks] {[type(c).__name__ for c in callbacks]}\n")

    experiment = load_experiment_from_hydra(cfg, task_name=task_name, callbacks=callbacks)
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()

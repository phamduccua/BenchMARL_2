"""
run_experiments.py
==================
Script chạy đa thuật toán × đa seed trên Kaggle / Colab.

Cách dùng
---------
Thay đổi ALGORITHMS, TASKS, SEEDS và EXPERIMENT_OVERRIDES ở phần CONFIG,
rồi chạy:

    python run_experiments.py

Hoặc trong notebook Kaggle/Colab:

    %run run_experiments.py

Khi chạy trên máy không có màn hình (headless), set HEADLESS = True
để tự động bọc xvfb-run.
"""

import subprocess
import sys
import itertools
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — chỉnh sửa ở đây
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = [
    "mappo",
    "ippo",
    "ippo_vi",
    "ippo_extragradient",
    "ippo_vi_no_norm",
    "ippo_vi_no_anchor",
    "maddpg",
    "iddpg",
    "masac",
    "isac",
]

TASKS = [
    "vmas/balance",
    # "vmas/navigation",
    # "vmas/simple_tag",
]

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Overrides Hydra áp dụng cho MỌI lần chạy
EXPERIMENT_OVERRIDES = [
    "experiment.loggers=[csv]",
    "experiment.render=false",
    "experiment.max_n_frames=3_000_000",
    "experiment.evaluation_interval=120_000",
    "experiment.on_policy_collected_frames_per_batch=6000",
    "experiment.on_policy_n_envs_per_worker=10",
    "experiment.on_policy_minibatch_size=32",   # batch size = 32
    "experiment.on_policy_n_minibatch_iters=45",
    "experiment.lr=0.00005",
    "experiment.checkpoint_interval=0",
]

# NashConv — tính mỗi evaluation step
NASHCONV_OVERRIDES = [
    "+nashconv.enable=true",
    "+nashconv.eval_interval=1",
    "+nashconv.br_updates=5",
    "+nashconv.br_episodes=4",
    "+nashconv.eval_episodes=5",
    "+nashconv.br_lr=3e-4",
]

# VI tau cho ippo_vi / ippo_vi_no_norm / ippo_vi_no_anchor
VI_OVERRIDES = [
    "algorithm.vi_tau=0.05",       # tau = 0.05
]

# True trên máy headless (Kaggle GPU, Colab)
HEADLESS = True

# Thư mục gốc của repo (chứa benchmarl/run.py)
REPO_ROOT = Path(__file__).parent.resolve()

# True = dừng hẳn khi 1 run bị lỗi; False = bỏ qua lỗi, chạy tiếp
FAIL_FAST = False

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _xvfb_available() -> bool:
    return subprocess.run(
        ["which", "xvfb-run"], capture_output=True
    ).returncode == 0


_VI_ALGOS = {"ippo_vi", "ippo_vi_no_norm", "ippo_vi_no_anchor"}


def _build_cmd(algorithm: str, task: str, seed: int) -> list[str]:
    extra = VI_OVERRIDES if algorithm in _VI_ALGOS else []
    base = [sys.executable, "benchmarl/run.py",
            f"algorithm={algorithm}",
            f"task={task}",
            f"seed={seed}",
            ] + EXPERIMENT_OVERRIDES + NASHCONV_OVERRIDES + extra

    if HEADLESS and _xvfb_available():
        base = ["xvfb-run", "-a"] + base

    return base


def run_single(algorithm: str, task: str, seed: int) -> int:
    """Chạy một thực nghiệm, trả về return-code."""
    cmd = _build_cmd(algorithm, task, seed)
    tag = f"[{algorithm} | {task} | seed={seed}]"
    print(f"\n{'='*60}")
    print(f"  START  {tag}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode == 0:
        print(f"\n✓  DONE   {tag}")
    else:
        print(f"\n✗  FAIL   {tag}  (exit={result.returncode})")

    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# HYDRA MULTIRUN — chạy tất cả combinations qua 1 lệnh duy nhất
# (nhanh hơn vì Hydra khởi tạo chỉ 1 lần)
# ─────────────────────────────────────────────────────────────────────────────

def run_multirun():
    """
    Dùng Hydra --multirun để chạy toàn bộ tổ hợp.
    Các run sẽ lần lượt trong cùng 1 process (không song song).
    """
    algo_str  = ",".join(ALGORITHMS)
    task_str  = ",".join(TASKS)
    seed_str  = ",".join(map(str, SEEDS))

    cmd = (
        (["xvfb-run", "-a"] if HEADLESS and _xvfb_available() else [])
        + [sys.executable, "benchmarl/run.py",
           "--multirun",
           f"algorithm={algo_str}",
           f"task={task_str}",
           f"seed={seed_str}",
           ] + EXPERIMENT_OVERRIDES + NASHCONV_OVERRIDES
           # Lưu ý: VI_OVERRIDES (vi_tau) chỉ có hiệu lực khi algorithm là VI.
           # Hydra sẽ bỏ qua field không tồn tại nếu dùng +algorithm.vi_tau=0.05

    )

    print("=" * 60)
    print("  HYDRA MULTIRUN")
    print(f"  algorithms : {ALGORITHMS}")
    print(f"  tasks      : {TASKS}")
    print(f"  seeds      : {SEEDS}")
    print(f"  total runs : {len(ALGORITHMS)*len(TASKS)*len(SEEDS)}")
    print("=" * 60)
    print(f"\nCMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL LOOP — chạy từng run độc lập (dễ debug hơn)
# ─────────────────────────────────────────────────────────────────────────────

def run_sequential():
    """Lần lượt chạy từng (algorithm, task, seed)."""
    combos = list(itertools.product(ALGORITHMS, TASKS, SEEDS))
    total  = len(combos)
    failed = []

    print(f"\nTotal runs: {total}  ({len(ALGORITHMS)} algos × {len(TASKS)} tasks × {len(SEEDS)} seeds)\n")

    for idx, (algo, task, seed) in enumerate(combos, 1):
        print(f"\n[{idx}/{total}]", end=" ")
        rc = run_single(algo, task, seed)
        if rc != 0:
            failed.append((algo, task, seed, rc))
            if FAIL_FAST:
                print("\nFAIL_FAST=True → dừng lại.")
                break

    print("\n" + "=" * 60)
    if failed:
        print(f"  {len(failed)} run(s) THẤT BẠI:")
        for algo, task, seed, rc in failed:
            print(f"    - {algo} | {task} | seed={seed}  (exit={rc})")
    else:
        print("  Tất cả runs hoàn thành thành công ✓")
    print("=" * 60)

    return len(failed)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sequential", "multirun"],
        default="sequential",
        help=(
            "sequential: mỗi run là 1 subprocess riêng (dễ debug).\n"
            "multirun:   dùng Hydra --multirun (1 process duy nhất, nhanh hơn)."
        ),
    )
    args = parser.parse_args()

    if args.mode == "multirun":
        sys.exit(run_multirun())
    else:
        sys.exit(run_sequential())

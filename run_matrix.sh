#!/bin/bash
set -e
cd /home/user/BenchMARL
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUTPUTS_DIR="/home/user/BenchMARL/outputs"

# ── Cấu hình chung cho matrix games ─────────────────────────────────────────
# Mỗi episode chỉ 1 bước nên dùng ít frame hơn so với simple_tag
BASE=(
    "experiment.loggers=[csv]"
    "experiment.render=false"
    "experiment.max_n_frames=500_000"

    "experiment.sampling_device=cuda"
    "experiment.train_device=cuda"
    "experiment.buffer_device=cuda"

    "experiment.on_policy_n_envs_per_worker=500"
    "experiment.on_policy_collected_frames_per_batch=10000"
    "experiment.on_policy_minibatch_size=1000"
    "experiment.on_policy_n_minibatch_iters=15"

    "experiment.off_policy_n_envs_per_worker=500"
    "experiment.off_policy_collected_frames_per_batch=10000"
    "experiment.off_policy_train_batch_size=1000"
    "experiment.off_policy_n_optimizer_steps=50"

    "experiment.evaluation_interval=20000"

    # NashConv: lấy π trực tiếp từ policy logits (1 forward pass, không cần sampling)
    "+nashconv.enable=true"
    "+nashconv.eval_interval=1"
)

# ── Hàm đóng gói kết quả ────────────────────────────────────────────────────
zip_results() {
    local zip_name=$1
    local zip_path="/home/user/results_${zip_name}.zip"

    echo "Đang gom và tái cấu trúc dữ liệu cho $zip_name..."

    local stage_dir=$(mktemp -d)
    local algo_dir="${stage_dir}/${zip_name}"
    mkdir -p "$algo_dir"

    cd "$OUTPUTS_DIR"

    while read -r yaml_file; do
        if grep -q "experiment.name=${zip_name}" "$yaml_file"; then
            local seed
            seed=$(grep "seed=" "$yaml_file" | tr -dc '0-9')
            local exp_dir
            exp_dir=$(dirname "$(dirname "$yaml_file")")
            local dest_dir="${algo_dir}/seed_${seed}"
            rm -rf "$dest_dir"
            cp -a "$exp_dir" "$dest_dir"
            echo "  + Đã map: $exp_dir ➔ ${zip_name}/seed_${seed}"
        fi
    done < <(find . -name "overrides.yaml")

    if [ "$(ls -A "$algo_dir")" ]; then
        (cd "$stage_dir" && zip -r -q "$zip_path" "$zip_name")
        echo "✔ Đã tạo file ZIP cấu trúc chuẩn: results_${zip_name}.zip"
    else
        echo "⚠️  Cảnh báo: Không tìm thấy dữ liệu nào cho $zip_name"
    fi

    rm -rf "$stage_dir"
    cd /home/user/BenchMARL
}

# ── Hàm chạy một thuật toán trên một task ───────────────────────────────────
# Dùng: run_algo <algo> <task_path> <zip_name> [extra overrides...]
#   algo      : tên algorithm trong benchmarl/conf/algorithm/
#   task_path : "matrix_game/rps" hoặc "matrix_game/matching_pennies"
#   zip_name  : tên dùng để đặt tên file ZIP và experiment.name
#   extra     : override tuỳ chọn thêm (vd: "algorithm.vi_tau=0.01")
run_algo() {
    local algo=$1
    local task_path=$2
    local zip_name=$3
    shift 3
    local extra=("$@")

    echo ""
    echo "============================================================"
    echo "🚀 TASK: ${task_path}  |  ALGO: $(echo "$zip_name" | tr 'a-z' 'A-Z')"
    echo "============================================================"

    local MAX_JOBS=3

    for seed in $(seq 0 9); do
        echo "  ➜ Đang chạy seed ${seed}/9 trong background..."

        python benchmarl/run.py \
            "algorithm=${algo}" \
            "task=${task_path}" \
            "seed=${seed}" \
            "+experiment.name=${zip_name}" \
            "${extra[@]}" \
            "${BASE[@]}" &

        if (( $(jobs -r -p | wc -l) >= MAX_JOBS )); then
            wait -n
        fi
    done

    wait
    echo "  ✔ Hoàn tất toàn bộ 10 seed cho $zip_name!"
    zip_results "$zip_name"
}

# ── Hàm chạy một thuật toán trên CẢ HAI task ────────────────────────────────
run_both() {
    local algo=$1
    local zip_name=$2
    shift 2
    local extra=("$@")

    run_algo "$algo" "matrix_game/rps"              "rps_${zip_name}"  "${extra[@]}"
    run_algo "$algo" "matrix_game/matching_pennies" "mp_${zip_name}"   "${extra[@]}"
}

# ── CLI ──────────────────────────────────────────────────────────────────────
usage() {
    echo ""
    echo "Cách dùng: $0 <task> <algo1> [algo2] ..."
    echo ""
    echo "  task  : rps | mp | both"
    echo "  algo  : ippo | mappo | ippo_vi | ippo_vi_tau001 | ippo_vi_tau01"
    echo "          ippo_vi_no_norm | ippo_vi_no_anchor | iql | qmix | vdn"
    echo "          ippo_extragradient | ippo_extragradient_self_adaptive"
    echo "          ippo_extragradient_self_adaptive_nu05"
    echo "          ippo_extragradient_self_adaptive_ms001 | ippo_extragradient_self_adaptive_ms05"
    echo "          ippo_projection_contraction | ippo_projection_contraction_alg2"
    echo ""
    echo "Ví dụ:"
    echo "  $0 both ippo mappo ippo_vi"
    echo "  $0 rps  ippo_vi_tau001 ippo_vi_tau01"
    echo "  $0 mp   ippo_vi"
    echo ""
}

if [ $# -lt 2 ]; then
    usage
    exit 1
fi

TASK_ARG=$1
shift

# Xác định task path từ tham số
case "$TASK_ARG" in
    rps)  TASKS=("matrix_game/rps") ;  PREFIXES=("rps") ;;
    mp)   TASKS=("matrix_game/matching_pennies") ; PREFIXES=("mp") ;;
    both) TASKS=("matrix_game/rps" "matrix_game/matching_pennies") ; PREFIXES=("rps" "mp") ;;
    *)
        echo "❌ Task không hợp lệ: '$TASK_ARG'. Dùng: rps | mp | both"
        usage
        exit 1
        ;;
esac

# Chạy từng algorithm được chỉ định
for algo in "$@"; do
    for i in "${!TASKS[@]}"; do
        task_path="${TASKS[$i]}"
        prefix="${PREFIXES[$i]}"

        case "$algo" in
            ippo_vi_tau001)
                run_algo "ippo_vi" "$task_path" "${prefix}_ippo_vi_tau001" \
                    "algorithm.vi_tau=0.01"
                ;;
            ippo_vi_tau01)
                run_algo "ippo_vi" "$task_path" "${prefix}_ippo_vi_tau01" \
                    "algorithm.vi_tau=0.1"
                ;;
            ippo_vi_tau005)
                run_algo "ippo_vi" "$task_path" "${prefix}_ippo_vi_tau005" \
                    "algorithm.vi_tau=0.05"
                ;;
            ippo_vi|ippo_vi_no_norm|ippo_vi_no_anchor)
                run_algo "$algo" "$task_path" "${prefix}_${algo}" \
                    "algorithm.vi_tau=0.05"
                ;;
            ippo_extragradient_self_adaptive_nu05)
                run_algo "ippo_extragradient_self_adaptive" "$task_path" "${prefix}_ippo_extragradient_self_adaptive_nu05" \
                    "algorithm.extra_nu=0.5"
                ;;
            ippo_extragradient_self_adaptive_ms001)
                run_algo "ippo_extragradient_self_adaptive" "$task_path" "${prefix}_ippo_extragradient_self_adaptive_ms001" \
                    "algorithm.extra_min_scale=0.01"
                ;;
            ippo_extragradient_self_adaptive_ms05)
                run_algo "ippo_extragradient_self_adaptive" "$task_path" "${prefix}_ippo_extragradient_self_adaptive_ms05" \
                    "algorithm.extra_min_scale=0.5"
                ;;
            ippo_extragradient_self_adaptive)
                run_algo "ippo_extragradient_self_adaptive" "$task_path" "${prefix}_ippo_extragradient_self_adaptive"
                ;;
            ippo_projection_contraction)
                run_algo "ippo_projection_contraction" "$task_path" "${prefix}_ippo_projection_contraction"
                ;;
            ippo_projection_contraction_alg2)
                run_algo "ippo_projection_contraction" "$task_path" "${prefix}_ippo_projection_contraction_alg2" \
                    "algorithm.viscosity_mode=alg2"
                ;;
            *)
                run_algo "$algo" "$task_path" "${prefix}_${algo}"
                ;;
        esac
    done
done

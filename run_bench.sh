# Đang chạy vi_tau=0.01

#!/bin/bash
set -e
cd /home/user/BenchMARL
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUTPUTS_DIR="/home/user/BenchMARL/outputs"

BASE=(
    "experiment.loggers=[csv]"
    "experiment.render=false"
    "experiment.max_n_frames=2_000_000"

    "experiment.sampling_device=cuda"
    "experiment.train_device=cuda"
    "experiment.buffer_device=cuda"
    
    "experiment.on_policy_n_envs_per_worker=200"
    "experiment.on_policy_collected_frames_per_batch=120000"
    "experiment.on_policy_minibatch_size=4096"
    "experiment.on_policy_n_minibatch_iters=15"

    "experiment.off_policy_n_envs_per_worker=200"
    "experiment.off_policy_collected_frames_per_batch=120000"
    "experiment.off_policy_train_batch_size=4096" 
    "experiment.off_policy_n_optimizer_steps=200"

    "experiment.evaluation_interval=240000"
    
    "+nashconv.enable=true"
    "+nashconv.eval_interval=1" 
    "+nashconv.br_episodes=1"
    "+nashconv.eval_episodes=1"
)

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
            local seed=$(grep "seed=" "$yaml_file" | tr -dc '0-9')
            local exp_dir=$(dirname $(dirname "$yaml_file"))
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
        echo "⚠️ Cảnh báo: Không tìm thấy dữ liệu nào cho $zip_name"
    fi
    
    rm -rf "$stage_dir"
    cd /home/user/BenchMARL
}

run_algo() {
    local algo=$1
    local zip_name=$2
    shift 2
    local extra=("$@")

    echo ""
    echo "============================================================"
    echo "🚀 BẮT ĐẦU CHẠY: $(echo $zip_name | tr 'a-z' 'A-Z')"
    echo "============================================================"

    MAX_JOBS=3

    for seed in $(seq 0 9); do
        echo "  ➜ Đang chạy seed ${seed}/9 trong background..."
        
        python benchmarl/run.py \
            "algorithm=${algo}" \
            "task=vmas/simple_tag" \
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

if [ $# -eq 0 ]; then
    echo "Cách dùng: $0 <algo1> [algo2] ..."
    echo "Ví dụ:    $0 ippo_vi ippo mappo ippo_projection_contraction ippo_projection_contraction_alg2"
    exit 1
fi

for algo in "$@"; do
    case "$algo" in
        ippo_vi_tau001)
            run_algo "ippo_vi" "ippo_vi_tau001" "algorithm.vi_tau=0.01"
            ;;
        ippo_vi_tau01)
            run_algo "ippo_vi" "ippo_vi_tau01" "algorithm.vi_tau=0.1"
            ;;
        ippo_vi|ippo_vi_no_norm|ippo_vi_no_anchor)
            run_algo "$algo" "$algo" "algorithm.vi_tau=0.05"
            ;;
        ippo_extragradient_self_adaptive_nu05)
            run_algo "ippo_extragradient_self_adaptive" "ippo_extragradient_self_adaptive_nu05" \
                "algorithm.extra_nu=0.5"
            ;;
        ippo_extragradient_self_adaptive_ms001)
            run_algo "ippo_extragradient_self_adaptive" "ippo_extragradient_self_adaptive_ms001" \
                "algorithm.extra_min_scale=0.01"
            ;;
        ippo_extragradient_self_adaptive_ms05)
            run_algo "ippo_extragradient_self_adaptive" "ippo_extragradient_self_adaptive_ms05" \
                "algorithm.extra_min_scale=0.5"
            ;;
        ippo_extragradient_self_adaptive)
            run_algo "ippo_extragradient_self_adaptive" "ippo_extragradient_self_adaptive"
            ;;
        ippo_projection_contraction)
            run_algo "ippo_projection_contraction" "ippo_projection_contraction"
            ;;
        ippo_projection_contraction_alg2)
            run_algo "ippo_projection_contraction" "ippo_projection_contraction_alg2" \
                "algorithm.viscosity_mode=alg2"
            ;;
        *)
            run_algo "$algo" "$algo"
            ;;
    esac
done
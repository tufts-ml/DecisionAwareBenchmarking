#!/usr/bin/env bash
# ====================================================================
# Benchmark re-run — Phase 2: Method-specific HP sweep
# ====================================================================
# For each method that has a tunable HP, sweeps that HP using the best
# (lr, batch) config found in Phase 1.
#
# Reads bench_p1_best.json (produced by collect_bench_p1.py) to pick
# the optimal (lr, batch) per method × task.
#
# Methods with HPs:
#   dfl      : dflalpha  ∈ {0.001, 0.01, 0.1*, 1.0, 10.0}
#   blackbox : lambd     ∈ {0.01, 0.05, 0.1*, 0.5, 1.0}
#   qptl     : tau       ∈ {0.1, 0.5, 1.0*, 5.0, 10.0}
#   listLTR  : tau       ∈ {0.1, 0.5, 1*, 5, 10}
#   lodl     : num_samples ∈ {100, 250, 500*, 1000, 2000}
#   perturb  : sigma     ∈ {0.1, 0.5, 1.0*, 2.0, 5.0}
#   perturb  : n_samples ∈ {5, 10*, 25, 50, 100}
#
# (* = Phase 1 default, included for completeness)
#
# Usage:
#   bash slurm/submit_bench_p2.sh --dry-run
#   bash slurm/submit_bench_p2.sh
#   bash slurm/submit_bench_p2.sh --problem knapsack
#   bash slurm/submit_bench_p2.sh --method dfl
# ====================================================================

set -e

DRY_RUN=false
PROB_FILTER=""
METHOD_FILTER=""
SEED_FILTER=""
# Site-specific: override via environment, e.g. PARTITION=gpu CONDA_ENV=myenv bash slurm/submit_bench_p2.sh
PARTITION="${PARTITION:-preempt}"
MEM="8G"
CONDA_ENV="${CONDA_ENV:-pco_bench_rhel7}"
SKIP_COMPLETED=true

# GPU cards with >=24 GB VRAM. shortestpath (ResNet18 + 10K warcraft images)
# OOMs on the 16 GB p100/t4 nodes, so it must be constrained to these.
GPU_24G_CONSTRAINT="rtx_6000-24G|rtx_a5000-24G|a100-40G|a100-80G|l40-48G|l40s-48G|rtx_6000_ada-48G|rtx_a6000-48G|h200-141G"
# Subset with >=40 GB, for dad on shortestpath (OOMs at 23.46 GiB on a 24 GB card).
GPU_40G_CONSTRAINT="a100-40G|a100-80G|l40-48G|l40s-48G|rtx_6000_ada-48G|rtx_a6000-48G|h200-141G"
BEST_JSON="bench_p1_best_val.json"   # val-selected HPs (test-leakage fix, Apr 2026). Override with --best-json to use pre-fix bench_p1_best.json.
MANIFEST_FILE="sweep_manifest_p2.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)           DRY_RUN=true; shift ;;
        --problem)           PROB_FILTER="$2"; shift 2 ;;
        --method)            METHOD_FILTER="$2"; shift 2 ;;
        --seed)              SEED_FILTER="$2"; shift 2 ;;
        --partition)         PARTITION="$2"; shift 2 ;;
        --best-json)         BEST_JSON="$2"; shift 2 ;;
        --no-skip-completed) SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Multi-seed model-init expansion (added 2026-05-21).
# Seed 0 keeps original prefix and skips no cells; seeds 1..9 append _s{N},
# pass --init_seed $((2023+N)), and exclude dad / energy / shortestpath /
# (perturb, budgetalloc) to match the Phase 1 sweep.
SEEDS=(0 1 2 3 4 5 6 7 8 9)

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ---- Load Phase 1 best configs ----
if [[ ! -f "$SCRIPT_DIR/$BEST_JSON" ]]; then
    echo "ERROR: $BEST_JSON not found. Run collect_bench_p1.py first."
    exit 1
fi

# ====================================================================
# Problem configuration (same as Phase 1)
# ====================================================================

PROBLEMS=(knapsack knapsack-real energy budgetalloc cubic bipartitematching portfolio asurv cook_county speed_humps sp_synth sp_planted pg_misspec shortestpath)

declare -A PROB_ARG
PROB_ARG[knapsack]=knapsack
PROB_ARG[knapsack-real]=knapsack
PROB_ARG[energy]=energy
PROB_ARG[budgetalloc]=budgetalloc
PROB_ARG[cubic]=cubic
PROB_ARG[bipartitematching]=bipartitematching
PROB_ARG[portfolio]=portfolio
PROB_ARG[asurv]=asurv
PROB_ARG[cook_county]=cook_county
PROB_ARG[speed_humps]=speed_humps
PROB_ARG[sp_synth]=sp_synth
PROB_ARG[sp_planted]=sp_planted
PROB_ARG[pg_misspec]=pg_misspec
PROB_ARG[shortestpath]=shortestpath

declare -A PROB_VERSION
PROB_VERSION[knapsack]=gen
PROB_VERSION[knapsack-real]=energy
PROB_VERSION[energy]=energy
PROB_VERSION[budgetalloc]=real
PROB_VERSION[cubic]=gen
PROB_VERSION[bipartitematching]=cora
PROB_VERSION[portfolio]=real
PROB_VERSION[asurv]=real
PROB_VERSION[cook_county]=real
PROB_VERSION[speed_humps]=real
PROB_VERSION[sp_synth]=synth
PROB_VERSION[sp_planted]=planted
PROB_VERSION[pg_misspec]=v3
PROB_VERSION[shortestpath]=warcraft

declare -A PROB_CONFIG
PROB_CONFIG[knapsack]=openpto/config/probs/knapsack_small.yaml
PROB_CONFIG[knapsack-real]=openpto/config/probs/knapsack-real.yaml
PROB_CONFIG[energy]=""
PROB_CONFIG[budgetalloc]=""
PROB_CONFIG[cubic]=""
PROB_CONFIG[bipartitematching]=""
PROB_CONFIG[portfolio]=""
PROB_CONFIG[asurv]=openpto/config/probs/asurv.yaml
PROB_CONFIG[cook_county]=openpto/config/probs/cook_county.yaml
PROB_CONFIG[speed_humps]=openpto/config/probs/speed_humps.yaml
PROB_CONFIG[sp_synth]=openpto/config/probs/sp_synth.yaml
PROB_CONFIG[sp_planted]=openpto/config/probs/sp_planted.yaml
PROB_CONFIG[pg_misspec]=openpto/config/probs/pg_misspec.yaml
PROB_CONFIG[shortestpath]=openpto/config/probs/shortestpath.yaml

declare -A INSTANCES
INSTANCES[knapsack]=400
INSTANCES[knapsack-real]=400
INSTANCES[energy]=400
INSTANCES[budgetalloc]=400
INSTANCES[cubic]=250
INSTANCES[bipartitematching]=20
INSTANCES[portfolio]=400
INSTANCES[asurv]=400             # silently ignored; dataset is fixed
INSTANCES[cook_county]=400       # silently ignored; dataset is fixed
INSTANCES[speed_humps]=400       # silently ignored; dataset is fixed
INSTANCES[sp_synth]=400
INSTANCES[sp_planted]=400
INSTANCES[pg_misspec]=400        # PG paper §4.1: n_train=200 + n_val=200 (val_frac forced to 0.5 in PGMisspec.__init__)
INSTANCES[shortestpath]=10000    # warcraft: 10K train images

declare -A TESTINSTANCES
TESTINSTANCES[knapsack]=200
TESTINSTANCES[knapsack-real]=200
TESTINSTANCES[energy]=200
TESTINSTANCES[budgetalloc]=200
TESTINSTANCES[cubic]=400
TESTINSTANCES[bipartitematching]=6
TESTINSTANCES[portfolio]=200
TESTINSTANCES[asurv]=200         # silently ignored
TESTINSTANCES[cook_county]=200   # silently ignored
TESTINSTANCES[speed_humps]=200   # silently ignored
TESTINSTANCES[sp_synth]=10000
TESTINSTANCES[sp_planted]=10000
TESTINSTANCES[pg_misspec]=10000
TESTINSTANCES[shortestpath]=1000

declare -A SOLVER_PTO
SOLVER_PTO[knapsack]=heuristic
SOLVER_PTO[knapsack-real]=gurobi
SOLVER_PTO[energy]=gurobi
SOLVER_PTO[budgetalloc]=neural
SOLVER_PTO[cubic]=heuristic
SOLVER_PTO[bipartitematching]=cvxpy
SOLVER_PTO[portfolio]=cvxpy
SOLVER_PTO[asurv]=heuristic
SOLVER_PTO[cook_county]=heuristic
SOLVER_PTO[speed_humps]=heuristic
SOLVER_PTO[sp_synth]=heuristic
SOLVER_PTO[sp_planted]=heuristic
SOLVER_PTO[pg_misspec]=heuristic
SOLVER_PTO[shortestpath]=heuristic

declare -A SOLVER_PNO
SOLVER_PNO[knapsack]=heuristic
SOLVER_PNO[knapsack-real]=gurobi
SOLVER_PNO[energy]=gurobi
SOLVER_PNO[budgetalloc]=neural
SOLVER_PNO[cubic]=heuristic
SOLVER_PNO[bipartitematching]=cvxpy
SOLVER_PNO[portfolio]=cvxpy
SOLVER_PNO[asurv]=heuristic
SOLVER_PNO[cook_county]=heuristic
SOLVER_PNO[speed_humps]=heuristic
SOLVER_PNO[sp_synth]=heuristic
SOLVER_PNO[sp_planted]=heuristic
SOLVER_PNO[pg_misspec]=heuristic
SOLVER_PNO[shortestpath]=heuristic

declare -A SOLVER_CVXPY
SOLVER_CVXPY[knapsack]=heuristic
SOLVER_CVXPY[bipartitematching]=cvxpy
SOLVER_CVXPY[portfolio]=cvxpy

get_walltime() {
    local prob="$1" group="$2"
    case "$group" in
        pto)
            case "$prob" in
                energy)             echo "10:00:00" ;;
                budgetalloc)        echo "5:00:00" ;;
                bipartitematching)  echo "5:00:00" ;;
                shortestpath)       echo "4:00:00" ;;   # GPU, ResNet18
                *)                  echo "1:30:00" ;;
            esac ;;
        pno)
            case "$prob" in
                energy)             echo "26:00:00" ;;
                budgetalloc)        echo "20:00:00" ;;   # high n_samples/stein_weight sweeps need many hours
                bipartitematching)  echo "5:00:00" ;;
                shortestpath)       echo "8:00:00" ;;   # GPU, ResNet18
                *)                  echo "2:00:00" ;;
            esac ;;
        lodl)
            case "$prob" in
                # ns2000 needs ~46h (measured scaling: ns500=11.9h, ns1000=22.9h,
                # ~2x per doubling) and TIMEOUT'd twice at 26h without reaching
                # epoch 1. 48h is the partition cap, so this is the only shot.
                energy)             echo "48:00:00" ;;
                # num_samples=2000 surrogate fit (serial) takes ~3.5h alone: seed 0
                # completed in 3:26 of a 4h limit, seeds 1-9 all TIMEOUT'd at 4h with
                # zero epochs (no ckpt is written until epoch 1, so they never resume).
                budgetalloc)        echo "10:00:00" ;;
                shortestpath)       echo "8:00:00" ;;   # GPU, ResNet18
                *)                  echo "4:00:00" ;;
            esac ;;
    esac
}

# Per-problem prediction model overrides
declare -A PRED_MODEL_ARGS
PRED_MODEL_ARGS[sp_synth]="--pred_model dense --n_layers 1"
PRED_MODEL_ARGS[sp_planted]="--pred_model dense --n_layers 1"
PRED_MODEL_ARGS[pg_misspec]="--pred_model dense --n_layers 1"
PRED_MODEL_ARGS[shortestpath]="--pred_model Resnet18"

# ====================================================================
# Method-specific HP definitions
# ====================================================================
# Each HP sweep entry: (method, hp_name, hp_values, yaml_key_or_flag,
#                       solver_group, method_problems, base_yaml)

# Methods that have tunable HPs in Phase 2
# Format stored below in arrays:
#   P2_METHODS: list of method names
#   For each method: HP name, values, solver group, scope

# dfl: dflalpha
DFL_VALS=(0.001 0.01 0.1 1.0 10.0)

# blackbox: lambd
BB_VALS=(0.01 0.05 0.1 0.5 1.0)

# qptl: tau  (knapsack/bipartitematching/portfolio only)
QPTL_VALS=(0.1 0.5 1.0 5.0 10.0)

# listLTR: tau
LISTLTR_VALS=(0.1 0.5 1 5 10)

# lodl: num_samples
LODL_VALS=(100 250 500 1000 2000)

# perturb: sigma (fixed n_samples=10)
PERTURB_SIGMA_VALS=(0.1 0.5 1.0 2.0 5.0)

# pg: sigma (finite difference width)
PG_SIGMA_VALS=(0.01 0.05 0.1 0.5 1.0)

# perturb: n_samples (using best sigma from sigma sweep, or default sigma=1.0 for now)
PERTURB_N_VALS=(5 10 25 50 100)

# ====================================================================
# Helper to read best config from JSON
# ====================================================================

get_best_lr() {
    local method="$1" prob="$2"
    python3 -c "
import json, sys
with open('$SCRIPT_DIR/$BEST_JSON') as f:
    d = json.load(f)
cfg = d.get('$method', {}).get('$prob', {})
print(cfg.get('lr', '1e-2'))
"
}

get_best_batch() {
    local method="$1" prob="$2"
    python3 -c "
import json, sys
with open('$SCRIPT_DIR/$BEST_JSON') as f:
    d = json.load(f)
cfg = d.get('$method', {}).get('$prob', {})
print(cfg.get('batch', 'default'))
"
}

# ====================================================================
# Helpers
# ====================================================================

prob_out_dir() { echo "${PROB_ARG[$1]}-${PROB_VERSION[$1]}"; }

# Cache active SLURM jobs once (full untruncated names, --format="%j").
_SLURM_ACTIVE_JOBS=$(squeue --user="$USER" --format="%j" --noheader 2>/dev/null || true)

is_completed() {
    local prob="$1" method="$2" prefix="$3"
    local out="$SCRIPT_DIR/saved_records/$(prob_out_dir $prob)/${method}/${prefix}/results.npy"
    [[ -f "$out" ]]
}

is_running() {
    # Returns true if a job with this exact name is already active in SLURM.
    local job_name="$1"
    echo "$_SLURM_ACTIVE_JOBS" | grep -qxF "$job_name"
}

has_checkpoint() {
    local prob="$1" method="$2" prefix="$3"
    local ckpt="$SCRIPT_DIR/saved_records/$(prob_out_dir $prob)/${method}/${prefix}/checkpoints/checkpoint_latest.pt"
    [[ -f "$ckpt" ]]
}

n_submitted=0
n_skipped=0

MANIFEST_TMP="$SCRIPT_DIR/.manifest_p2_tmp.jsonl"
: > "$MANIFEST_TMP"

add_manifest_entry() {
    local prob="$1" method="$2" prefix="$3" seed="${4:-0}"
    printf '{"prob":"%s","opt_model":"%s","prefix":"%s","seed":%d}\n' \
        "$prob" "$method" "$prefix" "$seed" >> "$MANIFEST_TMP"
}

# Main submit function
submit_p2_job() {
    local prob="$1" method="$2" hp_tag="$3" method_path="$4" solver_group="$5"
    local extra_args="${6:-}"
    local seed="${7:-0}"

    if [[ -n "$PROB_FILTER"   && "$prob"   != "$PROB_FILTER"   ]]; then return; fi
    if [[ -n "$METHOD_FILTER" && "$method" != "$METHOD_FILTER" ]]; then return; fi
    if [[ -n "$SEED_FILTER"   && "$seed"   != "$SEED_FILTER"   ]]; then return; fi

    # Multi-seed exclusions: seeds > 0 skip the slow / unstable cells.
    if [[ "$seed" -gt 0 ]]; then
        [[ "$method" == "dad" ]] && return 0
        [[ "$prob" == "energy" || "$prob" == "shortestpath" ]] && return 0
        [[ "$method" == "perturb" && "$prob" == "budgetalloc" ]] && return 0
    fi

    # Get best lr and batch from Phase 1
    local lr
    local batch_label
    lr=$(get_best_lr "$method" "$prob")
    batch_label=$(get_best_batch "$method" "$prob")

    # If not found in JSON, fall back to defaults
    [[ -z "$lr" || "$lr" == "None" ]] && lr="1e-2"
    [[ -z "$batch_label" || "$batch_label" == "None" ]] && batch_label="default"

    # Determine opt_name and batch_size flag
    local opt_name bs_flag=""
    if [[ "$batch_label" == "default" ]]; then
        # Use method-specific default
        case "$method" in
            spo|nce|pointLTR|pairLTR|listLTR) opt_name=sgd; bs_flag="--batch_size 1" ;;
            *) opt_name=gd ;;
        esac
    else
        opt_name=sgd
        local alt_bs=32
        [[ "$prob" == "bipartitematching" ]] && alt_bs=4
        bs_flag="--batch_size ${alt_bs}"
    fi

    # Determine solver
    local solver
    if [[ "$method" == "qptl" || "$method" == "cpLayer" ]]; then
        solver="${SOLVER_CVXPY[$prob]:-cvxpy}"
    elif [[ "$solver_group" == "pto" ]]; then
        solver="${SOLVER_PTO[$prob]}"
    else
        solver="${SOLVER_PNO[$prob]}"
    fi

    # Extra args: PtO on energy
    if [[ "$solver_group" == "pto" && "$prob" == "energy" ]]; then
        extra_args="$extra_args --skip_solver_eval"
    fi

    # Per-problem prediction model override
    local pred_model_args="${PRED_MODEL_ARGS[$prob]:-}"

    # GPU flag for problems that need it
    local gpu_flag=""
    local gpu_arg=""
    if [[ "$prob" == "shortestpath" ]]; then
        # See GPU_24G_CONSTRAINT: the 16 GB p100/t4 nodes CUDA-OOM on this task.
        # dad needs more still — it OOM'd at 23.46 GiB on an rtx_6000-24G
        # (pax177, 2026-08-01), so it requires a >=40 GB card.
        if [[ "$method" == "dad" ]]; then
            gpu_flag="--gres=gpu:1 --constraint=${GPU_40G_CONSTRAINT}"
        else
            gpu_flag="--gres=gpu:1 --constraint=${GPU_24G_CONSTRAINT}"
        fi
        gpu_arg="--gpu 0"
    fi

    local prob_arg="${PROB_ARG[$prob]}"
    local prob_cfg="${PROB_CONFIG[$prob]}"
    local cfg_flag=""
    [[ -n "$prob_cfg" ]] && cfg_flag="--config_path ${prob_cfg}"

    local instances="${INSTANCES[$prob]}"
    local testinstances="${TESTINSTANCES[$prob]}"
    local walltime
    walltime=$(get_walltime "$prob" "$solver_group")

    # LTR on shortestpath runs bs=1 → 10K steps/epoch; 8h walltime was hitting TIMEOUT.
    if [[ "$prob" == "shortestpath" && ( "$method" == "pointLTR" || "$method" == "pairLTR" || "$method" == "listLTR" ) ]]; then
        walltime="20:00:00"
    fi

    # Prefix encodes the hp value. Seed 0 keeps the original prefix so existing
    # results.npy are reused; seeds >0 append _s{N}.
    local prefix="bench_p2_${method}_${hp_tag}_${batch_label}_lr${lr}"
    [[ "$seed" -gt 0 ]] && prefix="${prefix}_s${seed}"

    # init_seed flag: seed 0 inherits the global --seed; seed N>0 reseeds model init.
    local init_seed_flag=""
    [[ "$seed" -gt 0 ]] && init_seed_flag="--init_seed $((2023 + seed))"

    local job_name="bp2_${prob}_${method}_${hp_tag}"
    [[ "$seed" -gt 0 ]] && job_name="${job_name}_s${seed}"

    add_manifest_entry "$prob" "$method" "$prefix" "$seed"

    if $SKIP_COMPLETED && is_completed "$prob" "$method" "$prefix"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip] ${prob} ${method} ${hp_tag}" || true
        return
    fi

    # Skip if already active in SLURM (running or pending) — avoids duplicate submission.
    if is_running "$job_name"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip/active] ${job_name}" || true
        return
    fi

    local cmd="python experiments/main_results.py \
        --problem ${prob_arg} \
        --opt_model ${method} \
        --solver ${solver} \
        --opt_name ${opt_name} \
        --lr ${lr} \
        --n_epochs 300 \
        --patience 40 \
        --instances ${instances} \
        --testinstances ${testinstances} \
        --seed 2023 \
        --n_ptr_epochs 0 \
        --prefix ${prefix} \
        --method_path ${method_path} \
        ${bs_flag} \
        ${cfg_flag} \
        ${pred_model_args} \
        ${gpu_arg} \
        ${init_seed_flag} \
        ${extra_args}"

    local log_file="$SCRIPT_DIR/logs/slurm/${job_name}_%j.out"

    # Per-problem memory override (shortestpath: ResNet18 + 10K images needs more RAM)
    local mem="$MEM"
    [[ "$prob" == "shortestpath" ]] && mem="32G"
    # LODL's 500-sample hessian + ResNet18 on 10K images blows past 32G (OOM killed at 33.5G)
    [[ "$prob" == "shortestpath" && "$method" == "lodl" ]] && mem="64G"

    if $DRY_RUN; then
        echo "[DRY-RUN] ${job_name}  (lr=${lr}, batch=${batch_label}, time=${walltime})"
        echo "  output: saved_records/$(prob_out_dir $prob)/${method}/${prefix}/results.npy"
        echo "  cmd: ${cmd}"
        return
    fi

    sbatch --job-name="$job_name" \
           --output="$log_file" \
           --partition="$PARTITION" \
           --cpus-per-task=2 \
           --mem="$mem" \
           --time="${walltime}" \
           --requeue \
           ${gpu_flag} \
           --wrap="
cd $SCRIPT_DIR
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
$cmd
"
    (( n_submitted++ )) || true
    echo "  [submit] ${job_name}"
}

# ====================================================================
# Helper: generate a method YAML for a given HP value on-the-fly
# Instead of creating many YAML files, we generate them at submit time.
# ====================================================================

ensure_yaml() {
    # Create a method YAML based on default.yaml but with one field changed.
    # Returns the path.
    local base_yaml="$1"
    local method="$2"
    local field="$3"
    local value="$4"
    local out_path="openpto/config/models/bench_p2_${method}_${field}_${value}.yaml"

    if [[ ! -f "$SCRIPT_DIR/$out_path" ]]; then
        python3 - <<PYEOF
import ruamel.yaml as yaml
with open("$SCRIPT_DIR/$base_yaml") as f:
    d = yaml.safe_load(f.read())
d["$method"]["$field"] = $value
with open("$SCRIPT_DIR/$out_path", "w") as f:
    yaml.dump(d, f)
PYEOF
    fi
    echo "$out_path"
}

# ====================================================================
# Phase 2 sweeps
# ====================================================================

echo "=== Benchmark Phase 2 sweep ==="
echo "Reading best configs from: $BEST_JSON"
echo "Partition: $PARTITION"
echo "Seeds:    ${SEEDS[*]}  (0 = existing runs; 1-9 → _s{N}, --init_seed 2024..2032)"
echo "Filters:  problem='${PROB_FILTER}' method='${METHOD_FILTER}' seed='${SEED_FILTER}'"
echo ""

BASE_YAML="openpto/config/models/default.yaml"
PERTURB_BASE="openpto/config/models/perturb_s1_n10.yaml"

DAD_STEIN_VALS=(0.1 0.5 1.0 2.0 5.0)

for seed in "${SEEDS[@]}"; do
    [[ -n "$SEED_FILTER" && "$seed" != "$SEED_FILTER" ]] && continue
    echo "===== Seed ${seed} ====="

    # ---- dfl: dflalpha ----
    if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "dfl" ]]; then
        echo "--- dfl: dflalpha sweep ---"
        for prob in "${PROBLEMS[@]}"; do
            for val in "${DFL_VALS[@]}"; do
                yaml_path=$(ensure_yaml "$BASE_YAML" "dfl" "dflalpha" "$val")
                submit_p2_job "$prob" "dfl" "alpha${val}" "$SCRIPT_DIR/$yaml_path" "pto" "" "$seed"
            done
        done
        echo ""
    fi

    # ---- blackbox: lambd ----
    if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "blackbox" ]]; then
        echo "--- blackbox: lambd sweep ---"
        for prob in "${PROBLEMS[@]}"; do
            for val in "${BB_VALS[@]}"; do
                yaml_path=$(ensure_yaml "$BASE_YAML" "blackbox" "lambd" "$val")
                submit_p2_job "$prob" "blackbox" "lam${val}" "$SCRIPT_DIR/$yaml_path" "pno" "" "$seed"
            done
        done
        echo ""
    fi

    # ---- qptl: tau ----
    if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "qptl" ]]; then
        echo "--- qptl: tau sweep ---"
        for prob in knapsack bipartitematching portfolio; do
            for val in "${QPTL_VALS[@]}"; do
                yaml_path=$(ensure_yaml "$BASE_YAML" "qptl" "tau" "$val")
                submit_p2_job "$prob" "qptl" "tau${val}" "$SCRIPT_DIR/$yaml_path" "pno" "" "$seed"
            done
        done
        echo ""
    fi

    # ---- listLTR: tau ----
    if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "listLTR" ]]; then
        echo "--- listLTR: tau sweep ---"
        for prob in "${PROBLEMS[@]}"; do
            for val in "${LISTLTR_VALS[@]}"; do
                yaml_path=$(ensure_yaml "$BASE_YAML" "listLTR" "tau" "$val")
                submit_p2_job "$prob" "listLTR" "tau${val}" "$SCRIPT_DIR/$yaml_path" "pno" "" "$seed"
            done
        done
        echo ""
    fi

    # ---- lodl: num_samples ----
    if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "lodl" ]]; then
        echo "--- lodl: num_samples sweep ---"
        for prob in "${PROBLEMS[@]}"; do
            for val in "${LODL_VALS[@]}"; do
                yaml_path=$(ensure_yaml "$BASE_YAML" "lodl" "num_samples" "$val")
                submit_p2_job "$prob" "lodl" "ns${val}" "$SCRIPT_DIR/$yaml_path" "lodl" "" "$seed"
            done
        done
        echo ""
    fi

    # ---- perturb: sigma sweep (fixed n_samples=10) ----
    if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "perturb" ]]; then
        echo "--- perturb: sigma sweep (n_samples=10) ---"
        for prob in "${PROBLEMS[@]}"; do
            for val in "${PERTURB_SIGMA_VALS[@]}"; do
                sig_tag="s${val//./p}"
                yaml_path=$(ensure_yaml "$PERTURB_BASE" "perturb" "sigma" "$val")
                submit_p2_job "$prob" "perturb" "${sig_tag}_n10" "$SCRIPT_DIR/$yaml_path" "pno" "" "$seed"
            done
        done
        echo ""

        # ---- perturb: n_samples sweep (fixed sigma=1.0) ----
        echo "--- perturb: n_samples sweep (sigma=1.0) ---"
        for prob in "${PROBLEMS[@]}"; do
            for val in "${PERTURB_N_VALS[@]}"; do
                yaml_path=$(ensure_yaml "$PERTURB_BASE" "perturb" "n_samples" "$val")
                submit_p2_job "$prob" "perturb" "s1p0_n${val}" "$SCRIPT_DIR/$yaml_path" "pno" "" "$seed"
            done
        done
        echo ""
    fi

    # ---- pg: sigma ----
    if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "pg" ]]; then
        echo "--- pg: sigma sweep ---"
        for prob in knapsack knapsack-real energy budgetalloc cubic bipartitematching portfolio asurv cook_county speed_humps sp_synth sp_planted pg_misspec; do
            for val in "${PG_SIGMA_VALS[@]}"; do
                yaml_path=$(ensure_yaml "$BASE_YAML" "pg" "sigma" "$val")
                submit_p2_job "$prob" "pg" "s${val//./p}" "$SCRIPT_DIR/$yaml_path" "pno" "" "$seed"
            done
        done
        echo ""
    fi

    # ---- dad: stein_weight ----
    if [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "dad" ]]; then
        echo "--- dad: stein_weight sweep ---"
        for prob in "${PROBLEMS[@]}"; do
            for val in "${DAD_STEIN_VALS[@]}"; do
                yaml_path=$(ensure_yaml "$BASE_YAML" "dad" "stein_weight" "$val")
                submit_p2_job "$prob" "dad" "sw${val//./p}" "$SCRIPT_DIR/$yaml_path" "pno" "" "$seed"
            done
        done
        echo ""
    fi
done

echo "Submitted: $n_submitted   Skipped (done): $n_skipped"

# Write manifest JSON (convert jsonl temp file → JSON array)
python3 -c "
import json
entries = [json.loads(l) for l in open('$MANIFEST_TMP') if l.strip()]
with open('$SCRIPT_DIR/$MANIFEST_FILE', 'w') as f:
    json.dump(entries, f, indent=2)
print(f'Manifest written to $MANIFEST_FILE ({len(entries)} entries)')
"
rm -f "$MANIFEST_TMP"

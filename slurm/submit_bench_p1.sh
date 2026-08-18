#!/usr/bin/env bash
# ====================================================================
# Benchmark re-run — Phase 1: LR × Batch sweep
# ====================================================================
# Runs all 15 methods × 13 tasks × 5 LRs × 2 batch configs.
# LRs match the original NeurIPS 2024 benchmark sweep: 1e-3, 5e-3, 1e-2, 5e-2, 1e-1.
# Each job uses --requeue so preempted jobs are automatically restarted;
# the ExpManager checkpoint/resume support picks up where it left off.
#
# Methods covered:
#   PtO:  mse, dfl, identity
#   PnO:  spo, nce, blackbox, pointLTR, pairLTR, listLTR, lodl, perturb
#         qptl, cpLayer  (knapsack/bipartitematching/portfolio only)
#
# Tasks: knapsack, knapsack-real, energy, budgetalloc, cubic,
#        bipartitematching, portfolio
#
# Usage:
#   bash slurm/submit_bench_p1.sh --dry-run
#   bash slurm/submit_bench_p1.sh
#   bash slurm/submit_bench_p1.sh --problem knapsack
#   bash slurm/submit_bench_p1.sh --method mse
#   bash slurm/submit_bench_p1.sh --method mse --problem knapsack
#   bash slurm/submit_bench_p1.sh --seed 1            # submit only init_seed 2024
#
# Multi-seed model-init expansion (added 2026-05-21):
#   SEEDS=(0..9); seed 0 = existing runs (prefix unchanged, no --init_seed);
#   seeds 1..9 append _s{N} to prefix and pass --init_seed $((2023 + N)).
#   Dataset is held fixed via --seed 2023 across all init_seeds (problem cache
#   lookup keys on rand_seed=2023, so all 10 seeds hit the same pickle).
# Exclusions for seeds > 0:
#   - dad (all problems)
#   - energy (all methods)
#   - shortestpath / warcraft (all methods)
#   - (perturb, budgetalloc)
# Seed 0 keeps full coverage so single-seed-only cells (the exclusions above)
# still appear in collect/figures with n_seeds=1.
# ====================================================================

set -e

DRY_RUN=false
PROB_FILTER=""
METHOD_FILTER=""
SEED_FILTER=""
# Site-specific: override via environment, e.g. PARTITION=gpu CONDA_ENV=myenv bash slurm/submit_bench_p1.sh
PARTITION="${PARTITION:-preempt}"
MEM="8G"
CONDA_ENV="${CONDA_ENV:-pco_bench_rhel7}"
SKIP_COMPLETED=true

# GPU cards with >=24 GB VRAM. shortestpath (ResNet18 + 10K warcraft images)
# OOMs on the 16 GB p100/t4 nodes, so it must be constrained to these.
GPU_24G_CONSTRAINT="rtx_6000-24G|rtx_a5000-24G|a100-40G|a100-80G|l40-48G|l40s-48G|rtx_6000_ada-48G|rtx_a6000-48G|h200-141G"
MANIFEST_FILE="sweep_manifest_p1.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)           DRY_RUN=true; shift ;;
        --problem)           PROB_FILTER="$2"; shift 2 ;;
        --method)            METHOD_FILTER="$2"; shift 2 ;;
        --seed)              SEED_FILTER="$2"; shift 2 ;;
        --partition)         PARTITION="$2"; shift 2 ;;
        --no-skip-completed) SKIP_COMPLETED=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs/slurm"

# ====================================================================
# Problem configuration
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
INSTANCES[knapsack-real]=400     # silently ignored; dataset is fixed
INSTANCES[energy]=400            # silently ignored; dataset is fixed
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
TESTINSTANCES[knapsack-real]=200 # silently ignored
TESTINSTANCES[energy]=200        # silently ignored
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

# ---- Per-problem solvers per method group ----
# PtO methods use fast solvers; PnO methods use the same (but gurobi for energy)
declare -A SOLVER_PTO   # MSE, DFL, Identity
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

declare -A SOLVER_PNO   # SPO, NCE, LTR, Blackbox, LODL, perturb
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

declare -A SOLVER_CVXPY  # QPTL, cpLayer — only valid for 3 tasks
SOLVER_CVXPY[knapsack]=heuristic    # qptl uses heuristic solver for knapsack
SOLVER_CVXPY[bipartitematching]=cvxpy
SOLVER_CVXPY[portfolio]=cvxpy

# ====================================================================
# Walltime table (by method group + problem)
# ====================================================================

# Walltime function: returns SLURM walltime for a given (prob, method_group)
# method_group: pto | pno | lodl | energy_pto | energy_pno
get_walltime() {
    local prob="$1" group="$2"
    case "$group" in
        pto)
            case "$prob" in
                energy)        echo "3:00:00" ;;   # skip_solver_eval; setup ~35min
                shortestpath)  echo "4:00:00" ;;   # GPU, ResNet18, 10K images
                *)             echo "1:30:00" ;;
            esac ;;
        pno)
            case "$prob" in
                energy)        echo "26:00:00" ;;  # Gurobi in training loop
                budgetalloc)   echo "4:00:00" ;;   # neural solver
                shortestpath)  echo "20:00:00" ;;  # GPU, ResNet18; LTR bs=1 is 10K steps/epoch
                *)             echo "2:00:00" ;;
            esac ;;
        lodl)
            case "$prob" in
                energy)        echo "26:00:00" ;;
                budgetalloc)   echo "4:00:00" ;;
                shortestpath)  echo "8:00:00" ;;   # GPU, ResNet18, 10K images
                *)             echo "4:00:00" ;;   # slow surrogate training
            esac ;;
    esac
}

# ====================================================================
# Batch configs
# ====================================================================
LRS=(1e-2 5e-3 1e-3 5e-2 1e-1)

# Batch configs: label → opt_name + batch_size flag
# Groups: default (full-batch gd) and alt (bs=32 sgd)
# Exception: SPO/NCE/LTR default is bs=1; their alt is bs=32
# Exception: bipartitematching alt is bs=4 (only 20 training instances)

# ====================================================================
# Method definitions
# ====================================================================
# Each method entry: name, solver_group (pto/pno/lodl),
#   default_opt, default_bs, alt_opt, alt_bs,
#   method_path, extra_args, problems (space-sep or "all")

# Format: METHOD_PROBLEMS[method] = "all" or space-separated problem names
declare -A METHOD_PROBLEMS
METHOD_PROBLEMS[mse]="all"
METHOD_PROBLEMS[mse_train]="all"
METHOD_PROBLEMS[mse_val]="all"
METHOD_PROBLEMS[dfl]="all"
METHOD_PROBLEMS[identity]="all"
METHOD_PROBLEMS[spo]="all"
METHOD_PROBLEMS[nce]="all"
METHOD_PROBLEMS[blackbox]="all"
METHOD_PROBLEMS[pointLTR]="all"
METHOD_PROBLEMS[pairLTR]="all"
METHOD_PROBLEMS[listLTR]="all"
METHOD_PROBLEMS[lodl]="all"
METHOD_PROBLEMS[perturb]="all"
METHOD_PROBLEMS[pg]="knapsack knapsack-real energy budgetalloc cubic bipartitematching portfolio asurv cook_county speed_humps sp_synth sp_planted pg_misspec"
METHOD_PROBLEMS[qptl]="knapsack bipartitematching portfolio"
METHOD_PROBLEMS[cpLayer]="knapsack bipartitematching portfolio"
METHOD_PROBLEMS[dad]="all"

declare -A METHOD_SOLVER_GROUP
METHOD_SOLVER_GROUP[mse]=pto
METHOD_SOLVER_GROUP[mse_train]=pto
METHOD_SOLVER_GROUP[mse_val]=pto
METHOD_SOLVER_GROUP[dfl]=pto
METHOD_SOLVER_GROUP[identity]=pto
METHOD_SOLVER_GROUP[spo]=pno
METHOD_SOLVER_GROUP[nce]=pno
METHOD_SOLVER_GROUP[blackbox]=pno
METHOD_SOLVER_GROUP[pointLTR]=pno
METHOD_SOLVER_GROUP[pairLTR]=pno
METHOD_SOLVER_GROUP[listLTR]=pno
METHOD_SOLVER_GROUP[lodl]=lodl
METHOD_SOLVER_GROUP[perturb]=pno
METHOD_SOLVER_GROUP[pg]=pno
METHOD_SOLVER_GROUP[qptl]=pno
METHOD_SOLVER_GROUP[cpLayer]=pno
METHOD_SOLVER_GROUP[dad]=pno

# Default batch config (benchmark default per method group)
declare -A METHOD_DEFAULT_OPT
METHOD_DEFAULT_OPT[mse]=gd
METHOD_DEFAULT_OPT[mse_train]=gd
METHOD_DEFAULT_OPT[mse_val]=gd
METHOD_DEFAULT_OPT[dfl]=gd
METHOD_DEFAULT_OPT[identity]=gd
METHOD_DEFAULT_OPT[spo]=sgd      # benchmark used bs=1
METHOD_DEFAULT_OPT[nce]=sgd
METHOD_DEFAULT_OPT[blackbox]=gd
METHOD_DEFAULT_OPT[pointLTR]=sgd
METHOD_DEFAULT_OPT[pairLTR]=sgd
METHOD_DEFAULT_OPT[listLTR]=sgd
METHOD_DEFAULT_OPT[lodl]=gd
METHOD_DEFAULT_OPT[perturb]=gd
METHOD_DEFAULT_OPT[pg]=gd
METHOD_DEFAULT_OPT[qptl]=gd
METHOD_DEFAULT_OPT[cpLayer]=gd
METHOD_DEFAULT_OPT[dad]=gd

declare -A METHOD_DEFAULT_BS     # only used when opt=sgd
METHOD_DEFAULT_BS[spo]=1
METHOD_DEFAULT_BS[nce]=1
METHOD_DEFAULT_BS[pointLTR]=1
METHOD_DEFAULT_BS[pairLTR]=1
METHOD_DEFAULT_BS[listLTR]=1

# Alt batch config (always sgd bs=32; bipartitematching uses bs=4)
METHOD_ALT_OPT=sgd   # all methods

# Method config YAML (--method_path)
declare -A METHOD_PATH
METHOD_PATH[mse]=openpto/config/models/default.yaml
METHOD_PATH[mse_train]=openpto/config/models/default.yaml
METHOD_PATH[mse_val]=openpto/config/models/default.yaml
METHOD_PATH[dfl]=openpto/config/models/default.yaml
METHOD_PATH[identity]=openpto/config/models/default.yaml
METHOD_PATH[spo]=openpto/config/models/default.yaml
METHOD_PATH[nce]=openpto/config/models/default.yaml
METHOD_PATH[blackbox]=openpto/config/models/default.yaml
METHOD_PATH[pointLTR]=openpto/config/models/default.yaml
METHOD_PATH[pairLTR]=openpto/config/models/default.yaml
METHOD_PATH[listLTR]=openpto/config/models/default.yaml
METHOD_PATH[lodl]=openpto/config/models/default.yaml
METHOD_PATH[perturb]=openpto/config/models/perturb_s1_n10.yaml   # sigma=1.0, n=10
METHOD_PATH[pg]=openpto/config/models/default.yaml
METHOD_PATH[qptl]=openpto/config/models/default.yaml
METHOD_PATH[cpLayer]=openpto/config/models/default.yaml
METHOD_PATH[dad]=openpto/config/models/default.yaml

# Per-problem prediction model overrides (empty = use default dense 3-layer)
declare -A PRED_MODEL_ARGS
PRED_MODEL_ARGS[sp_synth]="--pred_model dense --n_layers 1"      # linear model (SPO+ paper)
PRED_MODEL_ARGS[sp_planted]="--pred_model dense --n_layers 1"    # linear model (PG paper)
PRED_MODEL_ARGS[pg_misspec]="--pred_model dense --n_layers 1"    # linear model (PG paper §4.1)
PRED_MODEL_ARGS[shortestpath]="--pred_model Resnet18"             # ResNet18 (DPO paper)

# Extra args per method (e.g. --skip_solver_eval for PtO on energy)
# These are evaluated per (method, prob) at submit time

# ====================================================================
# Helpers
# ====================================================================

prob_out_dir() { echo "${PROB_ARG[$1]}-${PROB_VERSION[$1]}"; }

# Build set of job names currently RUNNING or PENDING in SLURM.
# Use --Format=name (capital F) to get full untruncated names.
# --format="%j" (lowercase, no width) gives full untruncated job names.
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
n_resuming=0

# Manifest: write one JSON-lines entry per call; merge to array at end.
MANIFEST_TMP="$SCRIPT_DIR/.manifest_p1_tmp.jsonl"
: > "$MANIFEST_TMP"   # truncate/create

add_manifest_entry() {
    local prob="$1" method="$2" prefix="$3" seed="${4:-0}"
    printf '{"prob":"%s","opt_model":"%s","prefix":"%s","seed":%d}\n' \
        "$prob" "$method" "$prefix" "$seed" >> "$MANIFEST_TMP"
}

submit_job() {
    local prob="$1" method="$2" batch_label="$3" lr="$4" seed="${5:-0}"

    if [[ -n "$PROB_FILTER"   && "$prob"   != "$PROB_FILTER"   ]]; then return; fi
    if [[ -n "$METHOD_FILTER" && "$method" != "$METHOD_FILTER" ]]; then return; fi
    if [[ -n "$SEED_FILTER"   && "$seed"   != "$SEED_FILTER"   ]]; then return; fi

    # Multi-seed exclusions: seeds > 0 skip the slow / unstable cells.
    # Seed 0 = existing single-seed runs; full coverage retained.
    if [[ "$seed" -gt 0 ]]; then
        [[ "$method" == "dad" ]] && return 0
        [[ "$prob" == "energy" || "$prob" == "shortestpath" ]] && return 0
        [[ "$method" == "perturb" && "$prob" == "budgetalloc" ]] && return 0
    fi

    # Check if this problem is in scope for this method
    local prob_list="${METHOD_PROBLEMS[$method]}"
    if [[ "$prob_list" != "all" ]]; then
        local found=false
        for p in $prob_list; do [[ "$p" == "$prob" ]] && found=true; done
        $found || return 0
    fi

    # Known-infeasible: perturb default-batch (full-batch gd, n_samples=10) on shortestpath
    # consumes ~22.6 GB on a 23.5 GB GPU and reliably OOMs. Skip entirely.
    if [[ "$prob" == "shortestpath" && "$method" == "perturb" && "$3" == "default" ]]; then
        return 0
    fi

    # Determine opt_name and batch_size
    local opt_name bs_flag=""
    if [[ "$batch_label" == "default" ]]; then
        opt_name="${METHOD_DEFAULT_OPT[$method]:-gd}"
        local default_bs="${METHOD_DEFAULT_BS[$method]:-}"
        [[ -n "$default_bs" ]] && bs_flag="--batch_size ${default_bs}"
    else
        # alt: sgd bs=32 (or bs=4 for bipartitematching)
        opt_name=sgd
        local alt_bs=32
        [[ "$prob" == "bipartitematching" ]] && alt_bs=4
        bs_flag="--batch_size ${alt_bs}"
    fi

    # Build prefix. Seed 0 keeps the original prefix so existing results.npy /
    # checkpoints are reused. Seeds >0 append _s{N}.
    local prefix="bench_p1_${method}_${batch_label}_lr${lr}"
    [[ "$seed" -gt 0 ]] && prefix="${prefix}_s${seed}"

    # init_seed flag: seed 0 inherits the global --seed; seed N>0 reseeds model init.
    local init_seed_flag=""
    [[ "$seed" -gt 0 ]] && init_seed_flag="--init_seed $((2023 + seed))"

    # Register in manifest (regardless of completion status)
    add_manifest_entry "$prob" "$method" "$prefix" "$seed"

    local job_name="bp1_${prob}_${method}_${batch_label}_lr${lr}"
    [[ "$seed" -gt 0 ]] && job_name="${job_name}_s${seed}"

    # Skip if already complete
    if $SKIP_COMPLETED && is_completed "$prob" "$method" "$prefix"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip] ${prob} ${method} ${batch_label} lr=${lr}" || true
        return
    fi

    # Skip if already active in SLURM (running or pending) — avoids duplicate submission
    if is_running "$job_name"; then
        (( n_skipped++ )) || true
        $DRY_RUN && echo "  [skip/active] ${job_name}" || true
        return
    fi

    local resuming=false
    has_checkpoint "$prob" "$method" "$prefix" && resuming=true

    # Determine solver
    local solver_group="${METHOD_SOLVER_GROUP[$method]}"
    local solver
    if [[ "$method" == "qptl" || "$method" == "cpLayer" ]]; then
        solver="${SOLVER_CVXPY[$prob]:-cvxpy}"
    elif [[ "$solver_group" == "pto" ]]; then
        solver="${SOLVER_PTO[$prob]}"
    else
        solver="${SOLVER_PNO[$prob]}"
    fi

    # Config path
    local prob_arg="${PROB_ARG[$prob]}"
    local prob_cfg="${PROB_CONFIG[$prob]}"
    local cfg_flag=""
    [[ -n "$prob_cfg" ]] && cfg_flag="--config_path ${prob_cfg}"

    # Instance counts
    local instances="${INSTANCES[$prob]}"
    local testinstances="${TESTINSTANCES[$prob]}"

    # Extra args: PtO methods on energy get --skip_solver_eval
    local extra_args=""
    if [[ "$solver_group" == "pto" && "$prob" == "energy" ]]; then
        extra_args="--skip_solver_eval"
    fi

    # MSE selection-criterion variants:
    #   mse_train: select on training MSE; no solver during training.
    #   mse_val:   select on validation MSE; no solver during training.
    # --opt_model carries the variant name (registered in utils_conf/wrapper_loss
    # as aliases for MSE); ExpManager dispatches on --selection_signal.
    if [[ "$method" == "mse_train" ]]; then
        extra_args="--selection_signal train_mse --skip_solver_eval"
    elif [[ "$method" == "mse_val" ]]; then
        extra_args="--selection_signal val_mse --skip_solver_eval --solver_valfreq 0"
    fi

    # Per-problem prediction model override
    local pred_model_args="${PRED_MODEL_ARGS[$prob]:-}"

    # GPU flag for problems that need it (e.g. warcraft shortestpath with ResNet18)
    local gpu_flag=""
    local gpu_arg=""
    if [[ "$prob" == "shortestpath" ]]; then
        # ResNet18 + 10K warcraft images needs >16 GB of VRAM. Landing on a
        # p100-16G / t4-16G node CUDA-OOMs ~2h in (6 jobs lost this way on
        # pax077, 2026-07-30), so constrain to cards with >=24 GB.
        gpu_flag="--gres=gpu:1 --constraint=${GPU_24G_CONSTRAINT}"
        gpu_arg="--gpu 0"
    fi

    # Walltime
    local walltime
    walltime=$(get_walltime "$prob" "$solver_group")

    local method_path="${METHOD_PATH[$method]}"

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
        local resume_tag=""
        $resuming && resume_tag=" [RESUME]"
        echo "[DRY-RUN]${resume_tag} ${job_name}  (time=${walltime})"
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
    $resuming && (( n_resuming++ )) || true
    echo "  [submit] ${job_name}$(${resuming} && echo ' [RESUME]' || echo '')"
}

# ====================================================================
# Main sweep
# ====================================================================

METHODS=(mse mse_train mse_val dfl identity spo nce blackbox pointLTR pairLTR listLTR lodl perturb pg qptl cpLayer dad)
BATCH_LABELS=(default alt)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

echo "=== Benchmark Phase 1 sweep ==="
echo "Methods:  ${METHODS[*]}"
echo "Problems: ${PROBLEMS[*]}"
echo "LRs:      ${LRS[*]}"
echo "Batches:  ${BATCH_LABELS[*]}"
echo "Seeds:    ${SEEDS[*]}  (0 = existing runs, no suffix; 1-9 → _s{N}, --init_seed 2024..2032)"
echo "Partition: $PARTITION"
echo "Filters:  problem='${PROB_FILTER}' method='${METHOD_FILTER}' seed='${SEED_FILTER}'"
echo ""

for seed in "${SEEDS[@]}"; do
    [[ -n "$SEED_FILTER" && "$seed" != "$SEED_FILTER" ]] && continue
    echo "===== Seed ${seed} ====="
    for prob in "${PROBLEMS[@]}"; do
        echo "--- ${prob} ---"
        for method in "${METHODS[@]}"; do
            for batch in "${BATCH_LABELS[@]}"; do
                for lr in "${LRS[@]}"; do
                    submit_job "$prob" "$method" "$batch" "$lr" "$seed"
                done
            done
        done
        echo ""
    done
done

echo "Submitted: $n_submitted   Resuming: $n_resuming   Skipped (done): $n_skipped"

# Write manifest JSON (convert jsonl temp file → JSON array)
python3 -c "
import json, sys
entries = [json.loads(l) for l in open('$MANIFEST_TMP') if l.strip()]
with open('$SCRIPT_DIR/$MANIFEST_FILE', 'w') as f:
    json.dump(entries, f, indent=2)
print(f'Manifest written to $MANIFEST_FILE ({len(entries)} entries)')
"
rm -f "$MANIFEST_TMP"

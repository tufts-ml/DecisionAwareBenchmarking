# Reproducing the paper

Two tiers:

- **Tier 1** — regenerate the paper's figures and generated tables from the
  bundled result JSONs in `results/`. Minutes on a laptop; no cluster, no
  datasets, no GPU.
- **Tier 2** — rerun the training sweep that produced those JSONs. Roughly
  20,000 training runs; see [Compute](#compute).

All commands run from the repo root. Scripts read `results/*.json` and write
to `results/figures/` and `results/tables/` (created on demand).

## Tier 1: figures and tables

`make all` regenerates everything below (`make tables` + `make figures`). The
generated tables are byte-identical to the manuscript's files, except the two
tuning-benefit tables whose captions and group headers are hand-written in the
manuscript (every data row matches). Regenerated PDFs differ from the
manuscript's files only in the PDF creation timestamp.

### Figures

The bump-chart scripts read `results/bench_p2_best_val.json`.

| Paper file (in the manuscript's `figures/`) | Command | Notes |
|---|---|---|
| `fig_bench_bump_not_recommended_bootstrap_cropped.pdf` | `python experiments/fig_bench_bump_not_recommended.py` | Emits plain + bootstrap variants; the paper uses the bootstrap one. The `_cropped` file is the same PDF with the manuscript's crop box set (trims the suptitle and legend); written with `pypdf`, skipped with a note if `pypdf` is not installed. |
| `fig_bench_bump_recommended_main_bootstrap_cropped.pdf` | `python experiments/fig_bench_bump_recommended.py` | Emits `recommended_{main,spatial}_{plain,bootstrap}`; `_cropped` as above. |
| `fig_bench_bump_recommended_spatial_bootstrap.pdf` | `python experiments/fig_bench_bump_recommended.py` | Same script. |
| `fig_bench_bump_bootstrap_portfolio.pdf` | `python experiments/fig_bench_bump_bootstrap.py` | The portfolio panel of the per-group outputs (appendix). |
| `synthetic_data.pdf` | `python experiments/fig_synthetic_data.py` | pg_misspec data-generating-process illustration (noise sign as in `openpto/problems/PGMisspec.py`). |

### Tables

| Paper file | Command | Input |
|---|---|---|
| `tabapp_tuning_benefit.tex`, `tab_tuning_benefit_summary.tex` | `python experiments/table_rank_counterfactual.py --subset notrec_rec_geo_spw --exclude mse_train mse_val --short_caption --name_suffix _paper --modes mean --use_cache` → `results/tables/notrec-rec-geo-spw_regret_change_mean_paper.tex` (per task) and `..._by_group_mean_paper.tex` | `rank_counterfactual_cache.json` |
| `error_per_problem.tex` | `python experiments/table_error_per_problem.py --exclude mse_train mse_val dad qptl cpLayer --problems knapsack knapsack-real energy budgetalloc cubic bipartitematching asurv cook_county speed_humps sp_synth pg_misspec shortestpath` | `loss_matrix.json` |
| `tabapp_geng_repro.tex` | `python experiments/table_geng_repro.py` | `geng_reproduction_data.json` |
| `tabapp_runtime.tex` | `python experiments/table_runtime_paper.py` | `runtimes_per_method_task.json` |
| `tabapp_fullgrid_pairs.tex`, `tabapp_fullgrid_cubic.tex` | `python experiments/table_fullgrid_paper.py` | `bench_fullgrid_best_val.json`, `bench_p2_best_val.json` |
| `tabapp_cubic_sense_fix.tex` | `python experiments/table_cubic_sense_fix.py` | `cubic_upstream_sense.json`, `bench_p2_best_val.json` |
| `tab_decision_problems.tex`, `tab_hyperparam_candidates.tex` | hand-maintained in the manuscript repo | — |

`make hp-table` (part of `make tables`) also writes
`results/tables/best_hyperparams.{csv,md,tex}`, the selected (LR, batch
regime, method-specific HP) per method x task; this is a repo deliverable,
not a paper table.

## Bundled result files

All in `results/`.

**`bench_p1_best_val.json`** — Phase-1 winners, nested `{method: {task: ...}}`.
Each cell records the `(lr, batch)` with the lowest 10-seed-mean validation
regret over the Phase-1 grid, with per-seed values and bookkeeping. Written by
`experiments/collect_bench_p1.py --metric val`. For `mse_val` the `val` field
is the validation prediction loss and for `mse_train` the training MSE (their
selection signals); it is not decision regret for those two methods.

**`bench_p2_best_val.json`** — Phase-2 winners; the file the figure scripts
read. Same nesting; each cell adds the method-specific hyperparameter
(`hp_name`, `hp_value`, `hp_tag`), `val`/`val_std`, `test`/`test_std`, the
per-seed `test_seed_values` the bootstrap CIs resample, and `fixed: {...}`
(the same cell at the method's default HP). Methods without a Phase-2 sweep
carry their Phase-1 cell. Written by `experiments/collect_bench_p2_val.py`.

**`loss_matrix.json`** — per (task, method): train/val/test prediction loss
and decision regret of the Phase-2 winner. Written by
`experiments/collect_loss_matrix.py --p2_best_json results/bench_p2_best_val.json`,
with `experiments/eval_test_pred.py` filling the per-run test prediction loss
between two collect passes.

**`rank_counterfactual_cache.json`** — per-task scan results (keys
`t|mean|<task>`, 12 paper tasks) so `table_rank_counterfactual.py --use_cache`
needs no training output. Without `--use_cache` the script rescans
`saved_records/`, which only exists after a Tier-2 rerun.

**`geng_reproduction_data.json`** — the 70 cells of the reproduction table:
Geng et al.'s published value, our single run under their seed, and the
10-seed mean/std, plus the summary statistics quoted in the appendix.

**`runtimes_per_method_task.json`** — per method x task: median active
wall-clock minutes per run of the selected configuration, run count and
CPU-hours of the sweep, reconstructed from per-epoch log timestamps with
scheduler preemption gaps excised.

**`bench_fullgrid_best_val.json`** — the joint {batch x LR x method-HP x seed}
grid of the joint-grid check for its eight (method, task) pairs, with the
full-grid validation argmax per pair.

**`cubic_upstream_sense.json`** — the Cubic Top-K column of the Phase-2
winners obtained with the upstream task declaration (see
[Cubic Top-K task declaration](#cubic-top-k-task-declaration)); the "released
code" column of `tabapp_cubic_sense_fix.tex`.

## Tier 2: retraining

### Fixed protocol (do not change)

Every run uses

```
--seed 2023  --n_epochs 300  --patience 40  --pred_model dense  --n_ptr_epochs 0
```

(`--pred_model Resnet18` for warcraft `shortestpath` only.)

### Benchmark fidelity table

Instance counts and solvers are part of the protocol; results are only
comparable to the paper if they are preserved exactly.

| Task | `--instances` | `--testinstances` | Solver | Per-task flags / notes |
|---|---|---|---|---|
| `knapsack` | 400 (default) | 200 (default) | `heuristic` (DP; validated equivalent to Gurobi, ~100x faster) | |
| `knapsack-real` | (don't pass) | (don't pass) | `gurobi` | `--problem knapsack --config_path openpto/config/probs/knapsack-real.yaml`; fixed real dataset |
| `energy` | (don't pass) | (don't pass) | `gurobi` | Fixed real dataset; the per-epoch evaluation solves the training and validation splits with Gurobi (see [Compute](#compute)) |
| `budgetalloc` | 400 (default) | 200 (default) | `neural` | |
| `cubic` | **250** | **400** | `heuristic` | Non-default counts, pass explicitly |
| `bipartitematching` | **20** | **6** | `heuristic` / `cvxpy` | Non-default counts, pass explicitly |
| `portfolio` | 400 (default) | 200 (default) | `cvxpy` | Reported as absolute (not relative) regret |
| `asurv` | (don't pass) | (don't pass) | `heuristic` (TopK) | Fixed real splits; 1338 locations, T=15/6/6 |
| `cook_county` | (don't pass) | (don't pass) | `heuristic` (TopK) | Fixed real splits; 1328 tracts, T=4/1/2 |
| `speed_humps` | (don't pass) | (don't pass) | `heuristic` (TopK) | Fixed real splits; 2107 tracts, T=5/2/4 |
| `sp_synth` | **400** | **10000** | `heuristic` (DAG DP on edge costs) | 5x5 grid (SPO+ paper DGP); `--pred_model dense --n_layers 1` |
| `sp_planted` | **400** | **10000** | `heuristic` (DAG DP) | Planted-arcs 5x5 grid (PG paper DGP); `--pred_model dense --n_layers 1` |
| `pg_misspec` | **400** | **10000** | `heuristic` (BinarySignSolver: z=1 iff Y<=0 under MINIMIZE) | PG paper Sec. 4.1 misspec experiment (variant 3); `--pred_model dense --n_layers 1`; `val_frac` forced to 0.5 in `PGMisspec.__init__` (n_train=200, n_val=200 per the paper) |
| `shortestpath` | **10000** | **1000** | `heuristic` (Dijkstra, 8-grid on vertex costs) | Warcraft 12x12 images; `--pred_model Resnet18`; **needs a GPU** (>=24 GB VRAM in our runs) |

Method applicability: `qptl`/`cpLayer` only on `knapsack` /
`bipartitematching` / `portfolio`; `pg` on everything except `shortestpath`;
all other methods on all 14 tasks.

### Selection signal

Every method selects checkpoints (and early-stops) on validation decision
regret (`--selection_signal val_regret`, the default), which means the solver
runs on the training and validation splits once per epoch. The two exceptions
are by design: `mse_train` (`--selection_signal train_mse`) and `mse_val`
(`--selection_signal val_mse --skip_solver_eval --solver_valfreq 0`) select on
prediction error and never call the solver during training.

### Phase 1: learning rate x batch regime

Grid per method x task: **5 learning rates** (1e-3, 5e-3, 1e-2, 5e-2, 1e-1,
the LR grid of Geng et al.) x **2 batch arms**:

- `default` — the Geng et al. regime: full-batch gradient descent
  (`--opt_name gd`), except SPO / NCE / pointLTR / pairLTR / listLTR, which
  use `sgd` with batch size 1.
- `alt` — minibatch `sgd` with batch size 32 (4 for `bipartitematching`).

Per-method semantics live in `batches_for` / `BATCH_LABELS` in
`experiments/collect_bench_p1.py` and `METHOD_DEFAULT_OPT` /
`METHOD_DEFAULT_BS` in `slurm/submit_bench_p1.sh`.

```bash
bash slurm/submit_bench_p1.sh --dry-run           # preview (SLURM)
bash slurm/submit_bench_p1.sh                     # submit everything
bash slurm/submit_bench_p1.sh --problem knapsack  # filters: --problem / --method / --seed
python experiments/collect_bench_p1.py --metric val   # -> results/bench_p1_best_val.json
```

The submit scripts target SLURM with `--requeue` plus checkpoint/resume
(SIGTERM handler in `openpto/expmanager/ExpManager.py`), so preempted jobs
restart from the last completed epoch. Site specifics (partition, conda env,
GPU constraint) are environment variables at the top of the scripts.

**Seeds.** Each cell runs 10 model-init seeds (`--init_seed 0..9`; the dataset
stays pinned to `--seed 2023`). Selection is by 10-seed-mean validation
regret; test regret is the 10-seed mean, with bootstrap CIs over
`test_seed_values` in the figures. Single-seed cells: `dad` (all tasks),
`energy` (all methods), `shortestpath` (all methods), and
(`perturb`, `budgetalloc`).

**Withdrawn batch arm (`BATCH_EXCLUDED_PAIRS`).** For `listLTR` x
`shortestpath` the `default` arm is withdrawn from selection: full-batch for an
LTR method means bs=1 over 10,000 warcraft images, 10,000 optimizer steps per
epoch, and every such run was walltime-killed before its final evaluation. The
cell is scored on the `alt` arm; its Phase-2 tau sweep was submitted at the
pre-withdrawal best LR, so `collect_bench_p2_val.py` falls back to the Phase-1
cell and marks it `"p2_fallback_to_p1": true`. `perturb`'s `default` arm is
infeasible on `shortestpath` (out of memory) and was never run.

### Phase 2: method-specific hyperparameter

Swept at the Phase-1-best `(lr, batch)` per method x task (read from
`results/bench_p1_best_val.json`). Grids (Phase-1 default in **bold**):

| Method | HP | Grid |
|---|---|---|
| `dfl` | `dflalpha` (MSE-blend weight) | 0.001, 0.01, **0.1**, 1.0, 10.0 |
| `blackbox` | `lambd` (interpolation step) | 0.01, 0.05, **0.1**, 0.5, 1.0 |
| `qptl` | `tau` | 0.1, 0.5, **1.0**, 5.0, 10.0 |
| `listLTR` | `tau` (softmax temperature) | 0.1, 0.5, **1**, 5, 10 |
| `lodl` | `num_samples` (surrogate-fit samples) | 100, 250, **500**, 1000, 2000 |
| `perturb` (DPO) | `sigma` (at n_samples=10) | 0.1, 0.5, **1.0**, 2.0, 5.0 |
| `perturb` (DPO) | `n_samples` (at sigma=1.0) | 5, **10**, 25, 50, 100 |
| `pg` | `sigma` (finite-difference width) | 0.01, 0.05, **0.1**, 0.5, 1.0 |
| `dad` | `stein_weight` | 0.1, 0.5, **1.0**, 2.0, 5.0 |

`mse`, `mse_train`, `mse_val`, `identity`, `spo`, `nce`, `pointLTR`, and
`pairLTR` have no method-specific hyperparameter and no Phase 2.

```bash
bash slurm/submit_bench_p2.sh --dry-run
bash slurm/submit_bench_p2.sh                     # reads results/bench_p1_best_val.json
python experiments/collect_bench_p2_val.py        # -> results/bench_p2_best_val.json
# loss matrix: collect (records run dirs), fill prediction losses from
# checkpoints, collect again
python experiments/collect_loss_matrix.py --p2_best_json results/bench_p2_best_val.json
python experiments/eval_test_pred.py              # per-run test_pred_loss.json
python experiments/collect_loss_matrix.py --p2_best_json results/bench_p2_best_val.json
```

Run-output prefixes under `saved_records/`: Phase 1
`bench_p1_{method}_{batch}_lr{lr}`, Phase 2
`bench_p2_{method}_{hp_tag}_{batch}_lr{lr}`; init seeds 1-9 append `_s{N}`.

### Known structural gaps

Cells without a number, for structural reasons; collectors and tables treat
them as missing:

- `shortestpath` x {`dfl`, `identity`}: no prediction loss. Both diverge to
  NaN at epoch 2 under the full-batch arm; the saved checkpoint's weights are
  NaN.
- `mse_train` / `mse_val`: no train/val regret, since they never run the
  solver during training.
- `dfl` x `energy`: no Phase-2 sweep; the reported cell is the Phase-1 pick at
  the default `dflalpha`.
- `pg` x `shortestpath`, `qptl`/`cpLayer` x non-LP problems: method not
  applicable.

### Compute

The two phases total roughly 20,000 training runs including the 10-seed
expansion (Phase 1 ~17.7k, Phase 2 ~4.4k), about 8,200 CPU-hours
(`tabapp_runtime.tex`). Each run is up to 300 epochs with early stopping
(patience 40). Most tasks train on CPU (~8 GB RAM per job) in minutes to a few
hours. The slow tail is per-epoch solver evaluation on the Gurobi tasks and
LODL's surrogate fitting at large `num_samples`. On `energy` the per-epoch
Gurobi evaluation of the training and validation splits costs roughly 1-4 min
per epoch for MSE and 5-20 min for DFL and Identity (it gets faster as
predictions improve), and single runs took up to 29 h; the submit scripts give
these cells 48 h and rely on checkpoint/requeue. Only warcraft `shortestpath`
needs a GPU (ResNet18 over 10,000 96x96 images; our jobs required >= 24 GB
VRAM). A single method x task cell (5 LRs x 2 arms x 10 seeds) is usually a
day-scale job array.

## Cubic Top-K task declaration

Every task declares in `init_API()['modelSense']` whether its objective is
maximised or minimised, and the decision-aware losses (SPO+, NCE, PG, DFL,
Identity, Blackbox, DAD, pairwise/listwise LTR) branch on that declaration;
MSE and pointwise LTR never read it, and DPO and LODL read it only off the
training path on this task. The upstream code
(`Thinklab-SJTU/PredictiveCO-Benchmark`, commit `f885426`) declares `cubic`
as MINIMIZE while its `TopKSolver` picks the K *largest* predicted values and
never consults the declaration, so on that task the sense-dependent losses
train with the wrong sign (SPO+'s loss, non-negative by construction, is
negative; the 160.4% regret Geng et al. report for SPO+ and NCE is the regret
of ranking items by minus their value). Decisions and regret are evaluated
correctly either way.

This repository declares MAXIMIZE (`openpto/problems/CubicTopK.py`);
`tests/test_solver_sense.py` checks that the declaration matches the solver's
direction and that the SPO+ loss is non-negative. The upstream declaration is
available with `--config_path openpto/config/probs/cubic_upstream_sense.yaml`
(a solver-kwarg override of `modelSense`); such runs share the
`saved_records/cubic-gen/<method>/<prefix>/` layout, so give them a distinct
`--prefix`. The manuscript's `tabapp_cubic_sense_fix.tex` compares, per
method, Geng et al.'s published value, this code under the upstream
declaration (`results/cubic_upstream_sense.json`), and this code with MAXIMIZE
declared (`results/bench_p2_best_val.json`); the reproduction table
`tabapp_geng_repro.tex` uses the upstream declaration for `cubic`, so that it
compares against the code as released.

## Tests

```bash
python -m pytest tests/ -q      # or: make test
```

`tests/test_solver_sense.py` (Cubic Top-K declaration vs solver direction,
SPO+ sign, upstream config) and `tests/test_bundled_results.py` (the bundled
JSONs are mutually consistent) always run. Two tests in
`tests/test_perturbed_softdecision.py` compare the DPO implementation against
the reference PyTorch port of Berthet et al.'s `perturbations` library and are
skipped unless that repo is cloned into the repo root:

```bash
git clone https://github.com/tuero/perturbations-differential-pytorch
```

# Reproducing the paper

Two tiers:

- **Tier 1** — regenerate the paper's figures and generated tables from the
  bundled result JSONs in `results/`. Minutes on a laptop, no cluster, no
  datasets, no GPU. The bundled JSONs hold the values the manuscript
  currently reports: the Cubic Top-K cells come from the re-run under the
  [sense correction](#cubic-top-k-sense-correction), and the `mse`/`dfl`/
  `identity` cells on `energy` come from the re-run with the per-epoch solver
  evaluation on (see [Selection signal](#selection-signal)).
- **Tier 2** — rerun the full training sweep that produced those JSONs.
  Roughly 22,000 training runs; see [Compute expectations](#compute-expectations).

All commands run **from the repo root**. Scripts read the JSONs from
`results/` and write to `results/figures/` and `results/tables/` (created on
demand).

## Tier 1: per-deliverable commands

### Figures

The bump-chart scripts read only `results/bench_p2_best_val.json`; the
rank-transition figure reads `results/bench_bump_rerun_data.json`. Regenerated
PDFs differ from the manuscript's files only in the PDF creation timestamp
(the drawn content streams are identical).

| Paper file (in the manuscript's `figures/`) | Command | Notes |
|---|---|---|
| `fig_bench_bump_not_recommended_bootstrap_cropped.pdf` | `python experiments/fig_bench_bump_not_recommended.py` | Emits plain + bootstrap variants; the paper uses the bootstrap one. The `_cropped` file is the bootstrap PDF with the manuscript's crop box set on the page (trims the suptitle and legend); the script writes it with `pypdf` (skipped with a note if `pypdf` is not installed). |
| `fig_bench_bump_recommended_main_bootstrap_cropped.pdf` | `python experiments/fig_bench_bump_recommended.py` | Emits `recommended_{main,spatial}_{plain,bootstrap}`; `_cropped` = same crop-box post-processing. |
| `fig_bench_bump_recommended_spatial_bootstrap.pdf` | `python experiments/fig_bench_bump_recommended.py` | Same script as above. |
| `fig_bench_bump_bootstrap_portfolio.pdf` | `python experiments/fig_bench_bump_bootstrap.py` | The portfolio panel of the per-group outputs (appendix). |
| `synthetic_data.pdf` | `python experiments/fig_synthetic_data.py` | pg_misspec data-generating-process illustration. Uses the benchmark's noise sign (Y = f* + zeta - 0.5, per `openpto/problems/PGMisspec.py`), correcting the sign in the originally submitted manuscript figure. |
| `fig_rank_transition_2col_7tasks.pdf` | `python experiments/fig_rank_transition.py` | Average-rank transition from the Geng et al. published table to our re-run (7 Geng tasks; a 5-task variant is emitted too). The file sits in the manuscript's `figures/` but is not included in the current build. |

### Tables

| Paper file | Command |
|---|---|
| `tabapp_tuning_benefit.tex` and `tab_tuning_benefit_summary.tex` | `python experiments/table_rank_counterfactual.py --subset notrec_rec_geo_spw --exclude mse_train mse_val --short_caption --name_suffix _paper --modes mean --use_cache` → per-task table `results/tables/notrec-rec-geo-spw_regret_change_mean_paper.tex` and by-group table `results/tables/notrec-rec-geo-spw_regret_change_by_group_mean_paper.tex`. Every data row (and the column-abbreviation header) matches the paper file exactly; the paper copies differ only in hand-written captions, labels, float placement and group-header wording. |
| `error_per_problem.tex` | `python experiments/table_error_per_problem.py --exclude mse_train mse_val dad qptl cpLayer --problems knapsack knapsack-real energy budgetalloc cubic bipartitematching asurv cook_county speed_humps sp_synth pg_misspec shortestpath` → `results/tables/error_per_problem.tex`, byte-identical to the paper file. |
| `tabapp_geng_repro.tex` | `python experiments/table_geng_repro.py` → `results/tables/tabapp_geng_repro.tex`, byte-identical. Reads `results/geng_reproduction_data.json`. Its cubic rows are the released-code values, by design (it checks that the published, mis-configured numbers reproduce). |
| `tabapp_runtime.tex` | `python experiments/table_runtime_paper.py` → `results/tables/tabapp_runtime.tex`, byte-identical. Reads `results/runtimes_per_method_task.json`; for `mse`/`dfl`/`identity` on `energy` every statistic (selected-configuration time, run count, CPU-hours) is that of the solver-evaluation-on re-run; the superseded legacy runs are not counted. |
| `tabapp_fullgrid_pairs.tex`, `tabapp_fullgrid_cubic.tex` | `python experiments/table_fullgrid_paper.py` → `results/tables/tabapp_fullgrid_{pairs,cubic}.tex`, byte-identical. Reads `results/bench_fullgrid_best_val.json` and `results/bench_p2_best_val.json`. |
| `tabapp_cubic_sense_fix.tex` | `python experiments/table_cubic_sense_fix.py` → `results/tables/tabapp_cubic_sense_fix.tex`, byte-identical. Reads `results/bench_p2_best_val_oldsense.json` (released code) and `results/bench_p2_best_val.json` (corrected). |
| `tab_decision_problems.tex`, `tab_hyperparam_candidates.tex` | **Hand-maintained** in the manuscript repo; not generated by any script here. |

`make tables` runs every generated-table command above (plus `make hp-table`;
the four appendix tables are also available as `make appendix-tables`);
`make figures` runs all five figure scripts; `make all` does both.

### Best-hyperparameter summary (repo deliverable, not a paper table)

```bash
python experiments/table_best_hyperparams.py     # or: make hp-table
```

emits `results/tables/best_hyperparams.{csv,md,tex}`: the selected
(LR, batch regime, method-specific HP) per method x task from the two
`*_best_val.json` files.

## Bundled result artifacts

One paragraph each; every file sits in `results/` and is committed. The
Cubic Top-K cells of the eight sense-dependent methods (`spo`, `nce`,
`identity`, `blackbox`, `pg`, `dfl`, `pairLTR`, `listLTR`) come from the
re-run under the [sense correction](#cubic-top-k-sense-correction); in our
records that re-run is its own prefix family, which is why those cells carry
`"family": "bench_cubicfix"` and `bench_cubicfix_p{1,2}_*` prefixes/run dirs
(a Tier-2 rerun with this repo's submit scripts produces them under the
ordinary `bench_p{1,2}_*` prefixes). Likewise the `mse`/`dfl`/`identity` cells
on `energy` carry `"family": "bench_energyvr"`: the re-run with the per-epoch
solver evaluation on ([Selection signal](#selection-signal)). Every other
cell is the original sweep.

**`bench_p1_best_val.json`** — Phase-1 winners. Nested
`{method: {task: {...}}}`; each cell records the val-regret-minimizing
`(lr, batch)` over the Phase-1 grid, with `val` (10-seed-mean validation
regret), per-seed `seed_values`, and bookkeeping fields (prefix, run dir,
seed counts). Produced by `experiments/collect_bench_p1.py --metric val`,
which selects on validation regret only — no test leakage. For `mse_val` on
every task `val` is the validation **prediction** loss the run selected on;
for `mse_train` it is the training MSE. In neither case is it decision
regret, and it is not comparable across methods.

**`bench_p2_best_val.json`** — Phase-2 winners; **the file every figure script
reads**. Same nesting; each cell adds the method-specific hyperparameter
(`hp_name`, `hp_value`, `hp_tag`), `prefix`/`run_dir`, `val`/`val_std` with
`seed_values`, `test`/`test_std` with `test_seed_values` (the 10-seed test
regrets that the bootstrap CIs resample), and `fixed: {...}` — the same cell
evaluated at the method's default HP. Methods with no Phase-2 sweep
(`mse`, `mse_train`, `mse_val`, `identity`, `spo`, `nce`, `pointLTR`,
`pairLTR`) carry Phase-1-fallback entries with no `hp_name`. Produced by
`experiments/collect_bench_p2_val.py`.

**`loss_matrix.json`** — per (task, method): train/val/test **prediction**
loss and **decision** regret for the val-best cell, plus its `run_dir`.
Input to `experiments/table_error_per_problem.py`. Produced by
`experiments/collect_loss_matrix.py --p2_best_json results/bench_p2_best_val.json`,
with `experiments/eval_test_pred.py` filling per-run `test_pred_loss.json`
files between two collect passes (the first pass records the run dirs the
eval script reads).

**`rank_counterfactual_cache.json`** — cached per-(seed-mode, task) scan
results (keys like `t|mean|<task>`) so
`table_rank_counterfactual.py --use_cache` needs no `saved_records/`
directory. Keys exist for all 12 paper tasks in `mean` mode (the mode the
paper tables use) plus `best`/`seed0` for five of the six not-recommended
tasks: only the `mean`-mode scans of `cubic` (after the sense correction)
and `energy` (after the solver-evaluation re-run) were redone, so their
`best`/`seed0` keys are absent rather than stale. Without
`--use_cache` the script re-scans raw training output, which only exists
after a Tier-2 rerun.

**`bench_p2_best_val_oldsense.json`** — the Phase-2 winners *before* the
Cubic Top-K sense correction, i.e. the released-code values (it also still
holds the legacy energy cells). Identical to `bench_p2_best_val.json` except
the eight affected cubic cells and the three energy cells. Read only by
`experiments/table_cubic_sense_fix.py`, and only its cubic cells.

**`geng_reproduction_data.json`** — the 70 cells of the reproduction table
(`rows`: Geng et al.'s published value, our seed-0 value, 10-seed mean/std and
count, flag) plus the summary statistics (`stats`) and provenance (`meta`).
The three energy PtO cells are the val-regret re-run; the cubic cells are the
released-code values. Read by `experiments/table_geng_repro.py`.

**`runtimes_per_method_task.json`** — per method x task: median active
wall-clock minutes of the val-selected configuration's runs and the per-task
sweep totals, reconstructed from per-epoch log timestamps with preemption gaps
excised. For `mse`/`dfl`/`identity` on `energy` the statistics are those of
the 30-run solver-evaluation-on re-run (the superseded legacy runs are not
counted, so the totals are the compute behind the reported cells). Read by
`experiments/table_runtime_paper.py`.

**`bench_fullgrid_best_val.json`** — the joint {batch x LR x method-HP x seed}
grid for the eight (method, task) pairs of the joint-grid check, with the
full-grid val-argmax per pair; the cubic pairs were re-run under the sense
correction. Read by `experiments/table_fullgrid_paper.py`.

**`bench_bump_rerun_data.json`** — per task x method test regret in display
units (percent; absolute for portfolio), the input of the rank-transition
figure. Each cell is the seed-0 run of the val-selected configuration, read
from its results file by the original figure pipeline; the eight corrected
cubic cells were spliced in afterwards as the 10-seed means of
`bench_p2_best_val.json` (this is how the manuscript's figure file was built),
so for those cells only, the value is a seed average rather than a single
run. `tests/test_bundled_results.py` checks the splice.

## Tier 2: retraining

### Fixed protocol (do not change)

Every run in the benchmark uses:

```
--seed 2023  --n_epochs 300  --patience 40  --pred_model dense  --n_ptr_epochs 0
```

(`--pred_model Resnet18` for warcraft `shortestpath` only, see below.)

### Benchmark fidelity table

Instance counts and solvers are part of the protocol — results are only
comparable to the paper (and to each other) if these are preserved exactly.

| Task | `--instances` | `--testinstances` | Solver | Per-task flags / notes |
|---|---|---|---|---|
| `knapsack` | 400 (default) | 200 (default) | `heuristic` (DP; validated equivalent to Gurobi, ~100x faster) | |
| `knapsack-real` | (don't pass) | (don't pass) | `gurobi` | Alias: `--problem knapsack --config_path openpto/config/probs/knapsack-real.yaml` (prob_version `energy`); fixed real dataset |
| `energy` | (don't pass) | (don't pass) | `gurobi` | Fixed real dataset. Per-epoch Gurobi evaluation of the validation split costs ~41 s; the original sweep skipped it for the PtO methods (`mse`, `dfl`, `identity`), the bundled cells and the submit scripts use the solver evaluation (see [Selection signal](#selection-signal)) |
| `budgetalloc` | 400 (default) | 200 (default) | `neural` | |
| `cubic` | **250** | **400** | `heuristic` | Non-default counts — must be passed explicitly |
| `bipartitematching` | **20** | **6** | `heuristic` / `cvxpy` | Non-default counts — must be passed explicitly |
| `portfolio` | 400 (default) | 200 (default) | `cvxpy` | Reported as absolute (not relative) regret |
| `asurv` | (don't pass) | (don't pass) | `heuristic` (TopK) | Fixed real splits; 1338 locations, T=15/6/6 |
| `cook_county` | (don't pass) | (don't pass) | `heuristic` (TopK) | Fixed real splits; 1328 tracts, T=4/1/2 |
| `speed_humps` | (don't pass) | (don't pass) | `heuristic` (TopK) | Fixed real splits; 2107 tracts, T=5/2/4 |
| `sp_synth` | **400** | **10000** | `heuristic` (DAG DP on edge costs) | Synthetic 5x5 grid (SPO+ paper DGP); `--pred_model dense --n_layers 1` |
| `sp_planted` | **400** | **10000** | `heuristic` (DAG DP) | Planted-arcs 5x5 grid (PG paper DGP); `--pred_model dense --n_layers 1` |
| `pg_misspec` | **400** | **10000** | `heuristic` (BinarySignSolver: z=1 iff Y<=0 under MINIMIZE) | PG paper Sec. 4.1 misspec experiment (variant 3); `--pred_model dense --n_layers 1`; `val_frac` forced to 0.5 in `PGMisspec.__init__` → n_train=200, n_val=200 per the paper |
| `shortestpath` | **10000** | **1000** | `heuristic` (Dijkstra, 8-grid on vertex costs) | Warcraft 12x12 images; `--pred_model Resnet18`; **needs a GPU** (>=24 GB VRAM in our runs) |

Method applicability: `qptl`/`cpLayer` only on `knapsack` /
`bipartitematching` / `portfolio`; `pg` on everything except `shortestpath`;
all other methods on all 14 tasks.

### Selection signal

All methods select checkpoints (and early-stop) on **validation decision
regret** (`--selection_signal val_regret`, the default), except `mse_train`
(`--selection_signal train_mse`) and `mse_val` (`--selection_signal val_mse`
with `--skip_solver_eval --solver_valfreq 0`) — these two never run the solver
during training by design.

**PtO methods on `energy` — legacy runs vs the bundled cells.** The original
sweep ran `mse`, `dfl`, and `identity` on `energy` with `--skip_solver_eval`
(the per-epoch Gurobi solve of the 130-instance validation split costs ~41 s),
so those three cells selected checkpoints and learning rates on validation
*prediction* error. The bundled JSONs and the manuscript instead use a
protocol-identical re-run with the solver evaluation on (single seed 2023 like
every other `energy` cell, val-regret selection over the same 2-arm x 5-LR
grid, no Phase 2 — the legacy Phase-2 sweep of `dfl` was selected on
prediction error and is withdrawn, so the `dfl` cell is the Phase-1 pick at
the default `dflalpha`). In test relative regret (selected cell in
parentheses):

| Cell | Legacy (`--skip_solver_eval`, val-MSE selection) | Bundled (solver eval on, val-regret selection) | Geng et al. (single run) |
|---|---|---|---|
| `mse` x `energy` | 2.14% (alt, lr=1e-1) | 2.11% (default, lr=5e-2) | 1.79% |
| `dfl` x `energy` | 17.15% (default, lr=1e-3) | 4.60% (default, lr=1e-2) | 6.27% |
| `identity` x `energy` | 17.08% (default, lr=1e-3) | 4.15% (alt, lr=1e-2) | 5.69% |

Restricted to the `default` arm (the Geng et al. regime used by the
manuscript's reproduction table) the picks are 2.07 / 17.15 / 17.08% (legacy)
vs 2.11 / 4.60 / 3.73% (val regret). Same seed and loss give the same training
trajectory; only early stopping and checkpoint selection differ, so the gap is
a selection-signal effect (val prediction error preferred lr=1e-3 runs whose
regret never moved). Val-regret selection keeps checkpoints with much larger
prediction error for `dfl` and `identity` (visible in the energy block of
`error_per_problem.tex`), which is expected for losses that never see the
prediction error. The re-run's cells carry `"family": "bench_energyvr"` in
the JSONs; `tabapp_runtime.tex` reports the re-run's runtimes and counts only
its compute for these cells. The submit scripts
run the solver evaluation **on** for these cells
(`ENERGY_PTO_SKIP_SOLVER_EVAL=0`), with 48 h walltimes: in the re-run the
median epoch took ~2 min (`mse`), ~3.5 min (`identity`) and ~7 min (`dfl`)
with 3-5x node-to-node variation, and the longest runs needed ~20 h (`mse`,
`identity`) and ~29 h (`dfl`) of active time; jobs checkpoint every epoch and
resume on requeue. Set `ENERGY_PTO_SKIP_SOLVER_EVAL=1` to reproduce the legacy
runs instead (their values are the "Legacy" column above). The bundled
`results/bench_p1_best_val.json` holds the val-regret (lr, batch) picks for
these cells, so `submit_bench_p2.sh` sweeps `dflalpha` on `energy` at the
right cell.

### Phase 1: learning rate x batch regime

Grid per method x task: **5 learning rates** (1e-3, 5e-3, 1e-2, 5e-2, 1e-1 —
the same LR grid as Geng et al.) x **2 batch arms**:

- `default` — the original Geng et al. regime: full-batch gradient descent
  (`--opt_name gd`), except SPO / NCE / pointLTR / pairLTR / listLTR which used
  `sgd` with batch size 1.
- `alt` — minibatch `sgd` with batch size 32 (batch size 4 for
  `bipartitematching`).

Exact per-method semantics live in `batches_for` / `BATCH_LABELS` in
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
GPU constraint) are overridable via environment variables at the top of the
scripts.

**Seeds.** Each cell runs with 10 model-init seeds (`--init_seed 0..9`; the
dataset stays pinned to `--seed 2023`, so all seeds share one problem
instance). Selection is by **10-seed-mean validation regret**; test regret is
reported as the 10-seed mean, with bootstrap CIs over `test_seed_values` in
the figures. Exceptions that only ever have seed 0 (single-seed cells):
`dad` (all tasks), `energy` (all methods), `shortestpath` (all methods), and
(`perturb`, `budgetalloc`).

**Withdrawn batch arm (`BATCH_EXCLUDED_PAIRS`).** For `listLTR` x
`shortestpath` the `default` arm is withdrawn from selection: full-batch for
an LTR method means bs=1 over 10,000 warcraft images, i.e. 10,000 optimizer
steps per epoch, and every such run was walltime-killed before finishing (the
val-best one died at epoch 54/300 with no final test eval). The cell is
scored on the `alt` arm only. Downstream consequence: the Phase-2 tau sweep
for that cell was submitted at the pre-withdrawal best LR, so
`collect_bench_p2_val.py` falls back to the Phase-1 cell (tau at its config
default) and marks the entry `"p2_fallback_to_p1": true`. Relatedly,
`perturb`'s `default` (full-batch) arm is infeasible on `shortestpath`
(out-of-memory: n_samples x full batch of ResNet18 activations) and was never
run. See the comments around `BATCH_EXCLUDED_PAIRS` in
`experiments/collect_bench_p1.py`.

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
# loss matrix: collect first (records run dirs), then fill prediction losses
# from checkpoints, then collect again
python experiments/collect_loss_matrix.py --p2_best_json results/bench_p2_best_val.json
python experiments/eval_test_pred.py              # per-run test_pred_loss.json
python experiments/collect_loss_matrix.py --p2_best_json results/bench_p2_best_val.json
```

Run-output prefix conventions (under `saved_records/`):
Phase 1 `bench_p1_{method}_{batch}_lr{lr}`, Phase 2
`bench_p2_{method}_{hp_tag}_{batch}_lr{lr}`; init seeds 1-9 append `_s{N}`.

### Known structural gaps (not data loss)

Some cells have no number for structural reasons; collectors and tables treat
them as missing rather than zero:

- `energy` x {`mse`, `dfl`, `identity`} under the legacy protocol
  (`ENERGY_PTO_SKIP_SOLVER_EVAL=1`): no train/val regret, because
  `--skip_solver_eval` records no in-training decision regret. The bundled
  cells come from the re-run with the solver evaluation on and do have it
  (see [Selection signal](#selection-signal)).
- `shortestpath` x {`dfl`, `identity`}: no prediction loss. Both diverge to
  NaN at epoch 2 under the full-batch arm (`loss=inf` → `nan`, flat
  thereafter); the saved checkpoint's weights are NaN, so there is no finite
  loss to report.
- `mse_train` / `mse_val`: no train/val regret anywhere — they select on MSE
  and never run the solver during training.
- `pg` x `shortestpath`, `qptl`/`cpLayer` x non-LP problems: method not
  applicable.

### Compute expectations

The two phases total roughly **22,000 training runs** including the 10-seed
expansion (Phase 1 ~17.7k, Phase 2 ~4.4k). Each run is up to 300 epochs with
early stopping (patience 40). Most tasks train on CPU (~8 GB RAM per job);
typical runs take minutes to a few hours, with the slow tail dominated by
per-epoch solver evaluation (Gurobi tasks) and LODL's surrogate fitting at
large `num_samples`. With the default (non-legacy) protocol the PtO methods on
`energy` join that tail: the longest observed runs took ~20 h (`mse`,
`identity`) and ~29 h (`dfl`) of active time, versus the 3 h (Phase 1) / 10 h
(Phase 2) walltimes the legacy `ENERGY_PTO_SKIP_SOLVER_EVAL=1` runs fit in. Only warcraft `shortestpath` needs a GPU (ResNet18 over
10,000 96x96 images; our jobs required >= 24 GB VRAM). Budget on the order of
tens of thousands of CPU-hours for the full sweep; a single method x task cell
(5 LRs x 2 arms x 10 seeds) is usually a day-scale job array.

## Cubic Top-K sense correction

Every task declares in `init_API()['modelSense']` whether its objective is
maximised or minimised, and the decision-aware losses (SPO+, NCE, PG, DFL,
Identity, Blackbox, DAD, pairwise/listwise LTR) branch on that declaration.
MSE and pointwise LTR never read it; DPO and LODL read it only off the training
path on this task. Upstream
(`Thinklab-SJTU/PredictiveCO-Benchmark`, commit `f885426`) declared `cubic` as
MINIMIZE while its `TopKSolver` picks the K *largest* predicted values (a
maximisation) and never consults the declaration, so on this one task those
losses trained with the wrong sign: SPO+'s loss — non-negative by construction
— is negative from the first epoch, and most SPO+ runs (78 of 100 in Phase 1)
land at exactly 160.4% regret (rank the items by *minus* their value), the
160.408% Geng et al. publish for SPO+ and NCE; the remaining runs sit on other
ranking attractors (39.6%), hence the 124% / 130% 10-seed means below.
Decisions and regret were always evaluated correctly; only the training signal
of the sense-dependent losses was inverted.

`openpto/problems/CubicTopK.py` now declares MAXIMIZE (a one-line change);
`tests/test_solver_sense.py` pins it (declared sense == direction the solver
optimises; SPO+ loss non-negative). `experiments/audit_solver_sense.py` applies
the same direction check to every task's solver
(`python experiments/audit_solver_sense.py [problem ...]`; `energy` needs a
Gurobi licence, `shortestpath` the Warcraft data): after the fix every task's
declared sense matches its solver (portfolio is reported `UNCLEAR` by the
heuristic, see the script's docstring). The upstream behaviour stays
reproducible with `--config_path openpto/config/probs/cubic_legacy_minsense.yaml`
(a solver-kwarg override of `modelSense`). Legacy and corrected runs share the
same `saved_records/cubic-gen/<method>/<prefix>/` layout and the same cached
`saved_problems/CubicTopK/*.pkl` (the sense lives in a method, not the pickle),
so give legacy runs a distinct `--prefix`.

Re-running the full two-phase protocol for the affected methods on `cubic`
(1,200 runs) gives, in 10-seed-mean test relative regret:

| Method | Released code (`bench_p2_best_val_oldsense.json`) | Corrected (`bench_p2_best_val.json`) | Note |
|---|---|---|---|
| SPO+ | 124.15 ± 58.37 | **0.01 ± 0.01** | now ahead of MSE (0.08) |
| NCE | 130.35 ± 46.53 | **29.65 ± 11.82** | learns the ranking; prediction scale unbounded |
| pr-LTR | 4.17 ± 5.49 | **0.09 ± 0.09** | |
| L-LTR | 0.22 ± 0.21 | **0.07 ± 0.07** | tau inert on this task |
| DFL | 8.61 ± 4.41 | **109.02 ± 65.35** | self-objective losses lock in the initial ranking |
| Identity / Blackbox | 14.75 ± 1.44 | **106.70 ± 64.14** | same mechanism (Blackbox ≡ Identity on linear objectives) |
| PG | 33.78 ± 17.59 | **94.81 ± 69.80** | same mechanism |
| MSE, pt-LTR, DPO, LODL | unchanged | unchanged | sense-independent (DPO and LODL re-runs bit-identical) |
| DAD | 22.36 (single seed) | not re-run | reads the sense; not in the paper's method set |

The bundled `results/*.json` hold the corrected values for the eight affected
`cubic` cells (every other cell is unaffected); the pre-correction Phase-2
winners are kept in `results/bench_p2_best_val_oldsense.json` for the
comparison table (`experiments/table_cubic_sense_fix.py` →
`tabapp_cubic_sense_fix.tex`). The reproduction table keeps the released-code
cubic values on purpose.

## Optional test dependency

`tests/test_solver_sense.py` (Cubic Top-K declared sense vs. solver direction,
SPO+ sign, legacy config) and `tests/test_bundled_results.py` (the bundled
JSONs are mutually consistent and carry the corrected cubic cells) are
self-contained and always run. Two tests in
`tests/test_perturbed_softdecision.py` compare our DPO implementation against
the reference PyTorch port of Berthet et al.'s `perturbations` library. They
are skipped unless the reference repo is present:

```bash
git clone https://github.com/tuero/perturbations-differential-pytorch
# clone into the repo root, keeping the directory name
python -m pytest tests/ -q      # or: make test
```

Without the clone, the suite passes with those two tests skipped; all
self-contained tests always run.

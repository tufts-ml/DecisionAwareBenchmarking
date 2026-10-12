# Regenerate the paper's figures and generated tables from the bundled result
# JSONs in results/ (no cluster, datasets or saved_records/ needed):
#
#   bench_p1_best_val.json, bench_p2_best_val.json   Phase-1/2 winners (figures, hp-table)
#   loss_matrix.json                                 error_per_problem.tex
#   rank_counterfactual_cache.json                   tuning-benefit tables
#   geng_reproduction_data.json                      tabapp_geng_repro.tex
#   runtimes_per_method_task.json                    tabapp_runtime.tex
#   bench_fullgrid_best_val.json                     tabapp_fullgrid_{pairs,cubic}.tex
#   cubic_upstream_sense.json                        tabapp_cubic_sense_fix.tex
#
# Run from the repo root: scripts read the JSONs from results/ and write to
# results/tables/ and results/figures/. Override the interpreter with e.g.
# `make PY=python3 tables`.
PY ?= python

.PHONY: all tables figures hp-table appendix-tables test

all: tables figures

# Paper tables:
#   results/tables/notrec-rec-geo-spw_regret_change_{,by_group_}mean_paper.tex
#     -> tabapp_tuning_benefit.tex / tab_tuning_benefit_summary.tex
#   results/tables/error_per_problem.tex (12 tasks x 12 methods)
#   results/tables/tabapp_{geng_repro,runtime,fullgrid_pairs,fullgrid_cubic,cubic_sense_fix}.tex
# plus the best-hyperparameter summary (see hp-table).
tables: hp-table appendix-tables
	$(PY) experiments/table_rank_counterfactual.py \
		--subset notrec_rec_geo_spw --exclude mse_train mse_val \
		--short_caption --name_suffix _paper --modes mean --use_cache
	$(PY) experiments/table_error_per_problem.py \
		--exclude mse_train mse_val dad qptl cpLayer \
		--problems knapsack knapsack-real energy budgetalloc cubic \
			bipartitematching asurv cook_county speed_humps sp_synth \
			pg_misspec shortestpath

# Appendix tables (reproduction check, runtimes, joint-grid check, cubic declaration)
appendix-tables:
	$(PY) experiments/table_geng_repro.py
	$(PY) experiments/table_runtime_paper.py
	$(PY) experiments/table_fullgrid_paper.py
	$(PY) experiments/table_cubic_sense_fix.py

# Best (LR, batch, method-HP) per method x task -> results/tables/ (.csv/.md/.tex)
hp-table:
	$(PY) experiments/table_best_hyperparams.py

# Paper figures -> results/figures/
figures:
	$(PY) experiments/fig_bench_bump_not_recommended.py
	$(PY) experiments/fig_bench_bump_recommended.py
	$(PY) experiments/fig_bench_bump_bootstrap.py
	$(PY) experiments/fig_synthetic_data.py

test:
	$(PY) -m pytest -q tests/

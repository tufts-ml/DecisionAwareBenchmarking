# Regenerate the paper's tables and figures from the bundled result JSONs
# (bench_p1_best_val.json, bench_p2_best_val.json, loss_matrix.json,
# rank_counterfactual_cache.json). No cluster or saved_records/ needed.
#
# Run from the repo root: scripts read the JSONs from ./ and write to
# results/tables/ and results/figures/.
#
# Override the interpreter with e.g. `make PY=python3 tables`.
PY ?= python

.PHONY: all tables figures hp-table test

all: tables figures

# Paper tables:
#   results/tables/notrec-rec-geo-spw_regret_change_{,by_group_}mean_paper.tex
#     -> tabapp_tuning_benefit.tex / tab_tuning_benefit_summary.tex
#   results/tables/error_per_problem.tex (12 tasks x 12 methods)
# plus the best-hyperparameter summary (see hp-table).
tables: hp-table
	$(PY) experiments/table_rank_counterfactual.py \
		--subset notrec_rec_geo_spw --exclude mse_train mse_val \
		--short_caption --name_suffix _paper --modes mean --use_cache
	$(PY) experiments/table_error_per_problem.py \
		--exclude mse_train mse_val dad qptl cpLayer \
		--problems knapsack knapsack-real energy budgetalloc cubic \
			bipartitematching asurv cook_county speed_humps sp_synth \
			pg_misspec shortestpath

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

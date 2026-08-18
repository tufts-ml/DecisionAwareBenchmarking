# Best hyperparameters per method × task

Selection: minimum 10-seed-mean validation decision regret over the Phase-1 grid (5 learning rates × 2 batch regimes) plus the Phase-2 method-specific HP grid. Cell format: `lr / batch[ / hp=value]`.

| Method | KS | KS-E | En | BA | Cu | BM | Pf | AS | CC | SH | SP-s | SP-p | PG-ms | SP-W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSE | 5e-2 / bs=32 | 5e-2 / bs=32 | 1e-1 / bs=32 | 1e-3 / bs=32 | 5e-3 / bs=32 | 5e-3 / bs=4 | 5e-3 / bs=32 | 1e-1 / full | 1e-1 / full | 1e-1 / full | 1e-1 / bs=32 | 1e-1 / bs=32 | 1e-1 / full | 5e-3 / bs=32 |
| MSE-tr | 5e-2 / bs=32 | 1e-1 / bs=32 | 1e-1 / bs=32 | 1e-3 / bs=32 | 1e-2 / bs=32 | 1e-3 / bs=4 | 1e-2 / bs=32 | 5e-2 / full | 1e-1 / full | 1e-1 / full | 5e-3 / bs=32 | 5e-3 / bs=32 | 5e-2 / full | 5e-3 / bs=32 |
| MSE-val | 5e-2 / full | 1e-1 / bs=32 | 1e-1 / bs=32 | 1e-3 / bs=32 | 1e-2 / bs=32 | 1e-3 / bs=4 | 5e-2 / bs=32 | 5e-2 / full | 1e-2 / full | 1e-1 / bs=32 | 1e-1 / bs=32 | 5e-2 / bs=32 | 5e-2 / bs=32 | 1e-2 / bs=32 |
| DFL | 1e-1 / full / α=10.0 | 5e-2 / bs=32 / α=1.0 | 1e-3 / full / α=0.001 | 1e-3 / full / α=1.0 | 1e-2 / full / α=10.0 | 1e-1 / bs=4 / α=10.0 | 1e-2 / full / α=0.1 | 1e-3 / full / α=1.0 | 1e-1 / full / α=0.01 | 1e-1 / full / α=10.0 | 5e-2 / full / α=10.0 | 1e-1 / bs=32 / α=10.0 | 5e-3 / bs=32 / α=10.0 | 1e-1 / full / α=0.001 |
| Identity | 1e-2 / bs=32 | 5e-3 / full | 1e-3 / full | 1e-2 / full | 5e-2 / full | 1e-3 / full | 1e-3 / bs=32 | 1e-2 / full | 1e-1 / full | 1e-1 / full | 1e-2 / bs=32 | 5e-3 / bs=32 | 5e-2 / bs=32 | 5e-2 / full |
| SPO+ | 5e-2 / bs=32 | 1e-1 / bs=32 | 1e-1 / bs=1 | 1e-3 / bs=1 | 5e-2 / bs=1 | 1e-2 / bs=1 | 5e-3 / bs=32 | 5e-2 / bs=32 | 1e-2 / bs=32 | 5e-2 / bs=32 | 5e-3 / bs=1 | 5e-2 / bs=32 | 1e-1 / bs=1 | 1e-3 / bs=32 |
| NCE | 5e-2 / bs=32 | 1e-3 / bs=32 | 5e-3 / bs=32 | 1e-3 / bs=32 | 1e-3 / bs=32 | 1e-2 / bs=4 | 1e-2 / bs=1 | 1e-1 / bs=32 | 5e-2 / bs=1 | 1e-3 / bs=1 | 1e-2 / bs=1 | 5e-2 / bs=32 | 1e-2 / bs=32 | 1e-2 / bs=1 |
| Blackbox | 1e-2 / full / λ=0.01 | 1e-1 / full / λ=0.01 | 1e-1 / full / λ=0.05 | 1e-3 / full / λ=0.05 | 5e-2 / full / λ=0.01 | 1e-3 / full / λ=0.01 | 5e-3 / bs=32 / λ=0.1 | 1e-2 / full / λ=0.01 | 1e-1 / full / λ=0.01 | 1e-1 / full / λ=0.01 | 5e-3 / bs=32 / λ=0.01 | 1e-3 / bs=32 / λ=0.01 | 1e-2 / full / λ=0.01 | 1e-2 / bs=32 / λ=0.01 |
| ptLTR | 1e-2 / bs=1 | 1e-1 / bs=32 | 1e-2 / bs=32 | 1e-3 / bs=1 | 5e-2 / bs=1 | 5e-2 / bs=1 | 5e-3 / bs=32 | 5e-3 / bs=32 | 1e-1 / bs=32 | 1e-3 / bs=1 | 5e-2 / bs=1 | 1e-2 / bs=1 | 1e-1 / bs=1 | 5e-3 / bs=32 |
| prLTR | 1e-2 / bs=32 | 5e-2 / bs=32 | 1e-2 / bs=32 | 5e-3 / bs=32 | 1e-2 / bs=1 | 1e-2 / bs=4 | 1e-3 / bs=1 | 5e-2 / bs=32 | 1e-1 / bs=32 | 1e-2 / bs=1 | 5e-2 / bs=1 | 5e-2 / bs=1 | 1e-2 / bs=1 | 1e-3 / bs=32 |
| lsLTR | 1e-2 / bs=32 / τ=0.1 | 5e-2 / bs=1 / τ=5 | 1e-2 / bs=32 / τ=0.5 | 5e-3 / bs=1 / τ=0.1 | 5e-2 / bs=1 / τ=5 | 5e-2 / bs=4 / τ=0.5 | 1e-3 / bs=1 / τ=0.5 | 5e-2 / bs=32 / τ=0.1 | 1e-1 / bs=32 / τ=0.1 | 1e-1 / bs=32 / τ=0.1 | 5e-2 / bs=32 / τ=0.5 | 5e-2 / bs=32 / τ=0.5 | 1e-1 / bs=1 / τ=0.1 | 1e-3 / bs=32 |
| LODL | 5e-2 / full / K=500 | 1e-1 / full / K=500 | 1e-1 / bs=32 / K=1000 | 1e-3 / full / K=2000 | 5e-2 / full / K=100 | 5e-3 / bs=4 / K=1000 | 1e-3 / bs=32 / K=1000 | 5e-2 / full / K=250 | 5e-2 / full / K=2000 | 1e-1 / full / K=250 | 1e-1 / bs=32 / K=2000 | 5e-2 / bs=32 / K=1000 | 1e-1 / bs=32 / K=500 | 1e-1 / bs=32 / K=250 |
| DPO | 5e-3 / full / M=100 | 5e-3 / full / M=100 | 5e-3 / bs=32 / σ=1.0 | 1e-2 / full / σ=0.5 | 5e-2 / bs=32 / σ=1.0 | 1e-2 / bs=4 / σ=1.0 | 5e-2 / bs=32 / M=50 | 1e-2 / full / M=25 | 1e-2 / full / M=100 | 1e-2 / full / σ=5.0 | 1e-2 / bs=32 / M=100 | 5e-3 / bs=32 / M=100 | 1e-1 / bs=32 / σ=1.0 | 1e-2 / bs=32 / M=5 |
| PG | 1e-2 / full / σ=0.01 | 1e-1 / bs=32 / σ=0.1 | 5e-2 / bs=32 / σ=0.1 | 1e-3 / bs=32 / σ=1.0 | 1e-2 / full / σ=0.5 | 5e-2 / full / σ=0.05 | 1e-1 / bs=32 / σ=0.5 | 1e-1 / bs=32 / σ=0.1 | 5e-2 / full / σ=0.1 | 1e-1 / full / σ=0.05 | 1e-1 / full / σ=1.0 | 1e-2 / bs=32 / σ=0.1 | 1e-1 / bs=32 / σ=0.05 | -- |
| QPTL | 5e-2 / bs=32 / τ=0.1 | -- | -- | -- | -- | 1e-3 / full / τ=0.1 | 5e-3 / full / τ=0.1 | -- | -- | -- | -- | -- | -- | -- |
| cpLayer | 1e-2 / full | -- | -- | -- | -- | 1e-3 / full | 1e-3 / bs=32 | -- | -- | -- | -- | -- | -- | -- |
| DAD | 1e-3 / full / w=1.0 | 1e-3 / full / w=0.1 | 1e-1 / full / w=5.0 | 1e-3 / full / w=2.0 | 1e-1 / bs=32 / w=0.5 | 1e-1 / bs=4 / w=1.0 | 1e-1 / bs=32 / w=0.1 | 5e-3 / full / w=0.1 | 5e-3 / full / w=0.5 | 1e-3 / full / w=0.5 | 5e-2 / bs=32 / w=5.0 | 1e-3 / full / w=5.0 | 1e-2 / full / w=0.1 | 1e-2 / full / w=2.0 |

**Legend**

- Batch regimes: `full` = full-batch gradient descent; `bs=n` = minibatch SGD with batch size n (bs=4 only on bipartite matching).
- HP symbols: α = DFL alpha (`dflalpha`), λ = Blackbox smoothing (`lambd`), τ = temperature (`tau`, QPTL/lsLTR), K = LODL samples (`num_samples`), σ = perturbation width (`sigma`, DPO/PG), M = DPO samples (`n_samples`), w = DAD Stein weight (`stein_weight`). Methods without a symbol have no Phase-2 HP sweep; their best config comes from Phase 1.
- `--` = method not applicable (QPTL/cpLayer only on KS/BM/Pf; PG not on SP-W).
- Tasks: KS = Knapsack (synth), KS-E = Knapsack (energy), En = Energy scheduling, BA = Budget allocation, Cu = Cubic Top-K, BM = Bipartite matching, Pf = Portfolio, AS = Whooping crane (asurv), CC = Cook County, SH = Speed humps, SP-s = Shortest path (synth), SP-p = Shortest path (planted), PG-ms = PG misspec, SP-W = Warcraft shortest path.

# Datasets

Each task reads its data from `openpto/data/<problem>/` — the training
entrypoint appends the problem name to `--data_dir` (default
`./openpto/data/`). Loaded problem instances are cached as pickles under
`saved_problems/`; pass `--loadnew True` after changing data on disk.

Three groups of tasks:

1. **Shipped with this repo** — `asurv/`, `cook_county/`, `speed_humps/`.
2. **Upstream benchmark zip** — `knapsack/`, `energy/`, `budgetalloc/`,
   `bipartitematching/`, `portfolio/`.
3. **Warcraft shortest path** — `shortestpath/`, downloaded separately.

The synthetic tasks — `knapsack` (gen), `cubic`, `sp_synth`, `sp_planted`,
`pg_misspec` — generate their data at load time and need nothing on disk.
(The `knapsack/` folder is only needed for the `knapsack-real` variant.)

## Shipped in-repo (committed, ~9 MB)

`asurv/`, `cook_county/`, and `speed_humps/` each contain fixed
spatiotemporal splits with the same schema:

```
train_x.csv  train_y.csv     # x: geoid, timestep, <feature columns>
valid_x.csv  valid_y.csv     # y: geoid, timestep, <target count>
test_x.csv   test_y.csv
```

- `asurv` — aerial-survey counts; 1338 locations, 15/6/6 timesteps.
- `cook_county` — Cook County (IL) Medical Examiner overdose deaths per census
  tract; 1328 tracts, 4/1/2 timesteps.
- `speed_humps` — NYC pedestrian-injury counts per census tract (speed-hump
  siting); 2107 tracts, 5/2/4 timesteps.

`speed_humps/raw/` additionally holds the two raw source CSVs
(`probabilities_census_tracts_year.csv`, built from NYC OpenData crash
records, and `centroid_table.csv`, tract centroids). The committed splits are
produced from them by `scripts/prep_speed_humps_data.py` (run from the repo
root); the script documents the temporal splits, feature columns, and
filtering.

## Upstream benchmark zip

Download from
[Google Drive](https://drive.google.com/file/d/10OQLzWS5b4EEEFjPc4YeVhxQ_021GoWW/view?usp=sharing)
(the data release accompanying Geng et al.'s NeurIPS 2024 benchmark) and unzip
into `./openpto/data/`. The files each loader expects:

| Folder | Files | Used by |
|---|---|---|
| `knapsack/` | `prices2013.dat` | `knapsack-real` (the synthetic `knapsack` needs no data) |
| `energy/` | `prices2013.dat`, `SchedulingInstances/` (load1/day*.txt), `energy_data.txt` | `energy` |
| `budgetalloc/` | `budget_allocation_data.pkl` | `budgetalloc` |
| `bipartitematching/` | `cora.cites`, `cora.content`, `cora_cites_metis.txt.part.27`, `cora.msubject.npy`, `cora_partition.pickle` | `bipartitematching` |
| `portfolio/` | `price_data_2004-01-01_2017-01-01_daily.pt` (plus the raw/intermediate CSVs it was built from) | `portfolio` |

The zip may contain additional folders (e.g. advertising) for upstream tasks
not included in this release; they are ignored.

Note on `portfolio`: if the `.pt` file is absent the loader falls back to
downloading raw prices via the legacy Quandl API, which is unlikely to still
work — use the file from the zip.

### Terms of use

The MIT license of this repository covers the **code only**. The datasets in
the upstream zip are derived from third-party sources and carry their original
terms, catalogued in Appendix C of
[Geng et al. (2024)](https://arxiv.org/abs/2311.07633):

- `knapsack/` and `energy/` — SEMO (Irish single electricity market operator)
  price data, publicly available regulated market data.
- `bipartitematching/` — the public Cora citation dataset.
- `budgetalloc/` — a processed derivative of Yahoo Webscope search-advertising
  data; **non-commercial academic use only**, subject to Yahoo's data-sharing
  terms.
- `portfolio/` — features/prices derived from the (discontinued) Quandl WIKI
  end-of-day dataset. The derived `.pt`/feature files are what the benchmark
  uses; treat the raw historical price CSVs as regeneration intermediates
  rather than data to redistribute.

The three datasets shipped in this repository (`asurv/`, `cook_county/`,
`speed_humps/`) are our own derivatives of public government data (see the
sections above).

## Warcraft shortest path (`shortestpath/`)

**Not** in the zip above. This is the Warcraft II 12x12 shortest-path dataset
of Vlastelica et al., *Differentiation of Blackbox Combinatorial Solvers*
(ICLR 2020) — the `warcraft_shortest_path_oneskin` archive distributed with
their official code release
([martius-lab/blackbox-differentiation-combinatorial-solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers);
follow the dataset link in that repo's README). Place the archive's `12x12/`
directory here:

```
openpto/data/shortestpath/12x12/
├── train_maps.npy              # (10000, 96, 96, 3) uint8 RGB map images
├── train_shortest_paths.npy    # (10000, 12, 12) optimal-path indicator
├── train_vertex_weights.npy    # (10000, 12, 12) true vertex costs
├── val_maps.npy                # (1000, ...) same trio for val
├── val_shortest_paths.npy
├── val_vertex_weights.npy
├── test_maps.npy               # (1000, ...) same trio for test
├── test_shortest_paths.npy
└── test_vertex_weights.npy
```

The loader (`openpto/problems/Shortestpath.py`) reads exactly these
`{split}_{maps,shortest_paths,vertex_weights}.npy` names; a file may
alternatively be sharded as `{name}_part*.npy` (the shards are stacked in
glob order).

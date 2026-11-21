# COEN874 term project: Expected SARSA, SARSA and Q-learning on CliffWalking across three slip levels
Note: config, results and tests are provided as config.zip, results.zip and tests.zip. Unzip them in the repository folder before running the scripts.

Research question: how does Expected SARSA compare with SARSA and Q-learning in terms of **online return**, and of
**safety** (cliff falls per 1,000 training steps), on the 4 x 12 CliffWalking task across three levels of
environment stochasticity (slip probability 0.1, 0.3 and 0.5)?

The `results/` folder contains the outputs of the full run reported in the paper (raw per-seed results, tables and
figures). `python analyze.py` regenerates every table and figure from the raw results in this folder; the tables
regenerate identically.

## 1. Environment and setup

* Environment: `CliffWalking-v1` layout (4 x 12, rewards -1 per step and -100 for the cliff, start 36, goal 47),
  reproduced in `cliff_mdp.py` as a Gymnasium-compatible class with our own slip wrapper.
* Slip model: the intended move happens with probability 1 - p, and each of the two perpendicular moves with
  probability p/2. The opposite move never happens. p = 2/3 reproduces Gymnasium's `is_slippery=True`; p = 0 is
  deterministic. Levels: Low 0.1, Medium 0.3, High 0.5.
* A cliff fall returns the agent to the start and the episode continues. Only reaching the goal terminates;
  the 500-step cap is truncation, so the TD target still bootstraps.
* Python 3.10 or newer. Install: `pip install -r requirements.txt`.

### Pinned versions (full run reported in the paper)

`requirements-pinned.txt` lists the exact package versions of the full run; `python versions.py` prints and rewrites
them for your own environment.
`results/run_config.json` also records Python, Gymnasium, NumPy, SciPy, pandas and Matplotlib versions.
PyTorch and Stable-Baselines3 are not used by this project.

| Package | Version |
|---|---|
| Python | 3.13.16 (Kaggle run) |
| Gymnasium | 1.2.0 |
| NumPy | 2.1.3 |
| SciPy, pandas, Matplotlib | 1.16.3, 2.3.3, 3.10.0 |
| pytest | 8.4.2 |
| PyTorch, Stable-Baselines3 | not used |

## 2. Run order

```
python -m pytest -q                      # unit tests (needs Gymnasium installed)
python experiment.py --smoke             # 2 seeds, 2,000 steps: end-to-end check, writes results_smoke/
python analyze.py --results results_smoke --boot 300     # smoke test of the analysis
python experiment.py --jobs 4            # full run: tuning on seeds 1000-1019, reporting on seeds 0-49
python analyze.py                        # all tables and figures into results/tables and results/figures
```

Settings are in `config/default.json` (20,000 training steps per run, epsilon 0.1, gamma 1, Q initialised to 0,
alpha grid of 7 values, 20 tuning seeds, 50 reporting seeds). Analysis settings are in its `analysis` block.

## 3. One command per figure or table

`python analyze.py --list` prints the names. Each item uses its own random stream, so running it alone gives
the same numbers as running everything.

| Item | Command | Output |
|---|---|---|
| Learning curves | `python analyze.py --only fig1` | `figures/fig1_learning_curves.pdf` |
| Per-seed distributions | `python analyze.py --only fig2` | `figures/fig2_distribution_per_seed.pdf` |
| Sensitivity to slip | `python analyze.py --only fig3` | `figures/fig3_sensitivity_to_slip.pdf` |
| Policy maps (behaviour) | `python analyze.py --only fig4` | `figures/fig4_policy_maps.pdf` |
| Safety versus online return | `python analyze.py --only fig5` | `figures/fig5_safety_vs_online_return.pdf` |
| Sensitivity to learning rate | `python analyze.py --only fig6` | `figures/fig6_sensitivity_to_alpha.pdf` |
| Summary by group | `python analyze.py --only summary` | `tables/summary_by_group.csv` |
| Statistical tests (pairwise and omnibus) | `python analyze.py --only tests` | `tables/statistical_tests_pairwise.csv`, `tables/statistical_tests_omnibus.csv` |
| Safety and return gap contrasts | `python analyze.py --only contrasts` | `tables/gap_contrasts_primary.csv` |
| Steps-to-threshold censoring | `python analyze.py --only censoring` | `tables/threshold_censoring.csv` |
| Main results table | `python analyze.py --only table_main` | `tables/table_main_results.csv` |
| Primary-outcome test table | `python analyze.py --only table_primary` | `tables/table_statistical_tests_primary.csv` |
| Hyperparameter table | `python analyze.py --only table_hyper` | `tables/table_hyperparameters.csv` |
| Environment and MDP table | `python analyze.py --only table_mdp` | `tables/table_mdp.csv` |
| DP references | `python analyze.py --only dp` | `tables/dp_references.csv` |

## 4. Outcomes and statistics

* **Safety** (primary): cliff falls per 1,000 training steps. **Online return** (primary): mean return per training
  episode. Because every step costs -1 and a fall costs an extra -99, each episode satisfies
  `return = -length - 99 x falls` (checked in the tests): online return combines path length and safety, so the
  two outcomes are linked by construction. `mean_episode_length` and `falls_per_episode` are reported to show
  where a difference in online return comes from.
* Secondary: final greedy performance (mean of the last 5 greedy-evaluation checkpoints, exact value from the MDP
  tables), gap to the DP optimum V*, area under the greedy-evaluation curve. Steps to within 10% of V* is reported
  only as a censoring count (runs that never reach it are not given a value).
* Unit of analysis: one score per seed. 95% percentile-bootstrap CIs (10,000 draws). Omnibus test across the three
  algorithms per metric and level: Welch's ANOVA (Kruskal-Wallis as robustness). Pairwise: Welch's t-test (primary)
  and Mann-Whitney U (robustness). Holm-Bonferroni correction (default family: the 9 pairwise comparisons within
  each metric; `--holm-family primary_pooled` pools the two primary metrics). Effect sizes: Hedges' g and the
  rank-biserial correlation. With NaN p-values Holm treats them as 1.
* Fair comparison: identical step budget; alpha chosen per algorithm and slip level on the tuning seeds
  (1000-1019) by mean online return, never on the reporting seeds (0-49). Evaluation is greedy, on a separate
  environment instance with its own random stream. Alpha is therefore tuned for online return; safety is measured at
  that alpha, and `fig6` shows both outcomes for every alpha.
* Reference points: uniform-random policy and the DP optimum V* (value iteration on the exact tables).
* Every training episode of every reporting run is logged in `results/episodes.csv.gz`; no seed is dropped.
  Distribution plots keep the axis readable but mark every seed beyond it on the axis edge with its count and value.

## 5. Reproducibility

* Seeds: each run derives its generators from (seed, algorithm, slip, role) with `numpy.random.SeedSequence`;
  the environment is seeded through `reset(seed=...)`; Python `random` and NumPy's global generator are also
  seeded per run, and PyTorch would be if it were imported (it is not used here).
* All hyperparameters are in `config/default.json`. Raw per-seed results: `results/per_seed.csv`,
  `results/episodes.csv.gz`, `results/raw_report.pkl`, `results/tuning_per_seed.csv`.
* The smoke test (`python experiment.py --smoke`) runs tuning and reporting end to end with 2 seeds.
* Tests: `tests/test_core.py` (environment tables against Gymnasium's own table at slip 0 and 2/3, slip model,
  TD targets, termination versus truncation, equal budgets, reproducibility, return identity) and
  `tests/test_stats.py` (Holm, Welch's ANOVA, effect sizes, bootstrap).
* Third-party code: Gymnasium (`gym.Env` and `spaces`), NumPy, SciPy, pandas, Matplotlib. The agents, MDP solver
  and analysis are our own implementations of the algorithms in Sutton and Barto (2018).

## 6. Running on Kaggle

1. Upload the zip as a dataset (or the notebook's file panel), then in a cell:
   `!cp -r /kaggle/input/<dataset-name>/coen874 /kaggle/working/ && cd /kaggle/working/coen874`
2. `!pip install -q -r requirements.txt` (Kaggle already has most packages; this keeps versions consistent).
3. `!python -m pytest -q`, then `!python experiment.py --smoke`, then `!python experiment.py --jobs 4`,
   then `!python analyze.py`, then `!python versions.py`.
4. Download `results/` (tables, figures, CSV files) and `requirements-pinned.txt`.
Use `%cd` rather than `!cd` in a notebook cell, because `!cd` does not persist between cells.

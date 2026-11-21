"""Run tuning (separate seeds) and reporting experiments.

  python experiment.py --smoke                 # 2 seeds, small budget, ~seconds
  python experiment.py                         # full run (config/default.json)
  python experiment.py --phase tune|report|all --out results

Outputs (in --out): references.json, tuning.csv, tuning_per_seed.csv, chosen_alpha.json, per_seed.csv,
episodes.csv.gz (every training episode of every reporting run), raw_report.pkl, run_config.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version

import gymnasium
import numpy as np

from agents import ALGOS, run_agent
from cliff_mdp import ENV_ID, references

SCORE_KEYS = ["online_return", "fall_rate_per_1k", "final_eval_exact", "final_eval_sim", "gap_vs_vstar",
              "auc_eval", "steps_to_threshold", "censored", "n_episodes", "total_falls"]


def _versions():
    out = {}
    for pkg in ("gymnasium", "numpy", "scipy", "pandas", "matplotlib", "pytest", "torch", "stable-baselines3"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _task(args):
    algo, slip, alpha, seed, cfg = args
    return run_agent(algo, slip, alpha, seed, cfg)


def _map(tasks, jobs):
    if jobs <= 1:
        return [_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(_task, tasks, chunksize=8))


def tune(cfg, out, jobs):
    """Choose alpha per (level, algorithm) by mean online return on the tuning seeds (never the reporting seeds).

    Every tuning run is also saved (tuning_per_seed.csv) so the learning-rate sensitivity figure
    can show online return AND cliff falls for every alpha.
    """
    rows, per_seed, chosen = [], [], {}
    for level, slip in cfg["slip_levels"].items():
        for algo in ALGOS:
            best = None
            for alpha in cfg["alpha_grid"]:
                seeds = range(cfg["tuning_seed_offset"], cfg["tuning_seed_offset"] + cfg["tuning_seeds"])
                res = _map([(algo, slip, alpha, s, cfg) for s in seeds], jobs)
                vals = np.array([r["scores"]["online_return"] for r in res])
                falls = np.array([r["scores"]["fall_rate_per_1k"] for r in res])
                for r in res:
                    per_seed.append({"level": level, "slip": slip, "algo": algo, "alpha": alpha, "seed": r["seed"],
                                     "online_return": r["scores"]["online_return"],
                                     "fall_rate_per_1k": r["scores"]["fall_rate_per_1k"]})
                rows.append({"level": level, "slip": slip, "algo": algo, "alpha": alpha,
                             "mean_online_return": vals.mean(), "sd": vals.std(ddof=1), "n": len(vals),
                             "mean_fall_rate_per_1k": falls.mean(), "sd_fall_rate_per_1k": falls.std(ddof=1)})
                if best is None or vals.mean() > best[1]:
                    best = (alpha, vals.mean())
            chosen[f"{level}|{algo}"] = best[0]
            print(f"  tuned {level:6s} {algo:15s} alpha={best[0]}  online_return={best[1]:.2f}")
    for name, data in (("tuning.csv", rows), ("tuning_per_seed.csv", per_seed)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    json.dump(chosen, open(os.path.join(out, "chosen_alpha.json"), "w"), indent=2)
    return chosen


def report(cfg, out, chosen, jobs):
    tasks, meta = [], []
    seeds = range(cfg["reporting_seed_offset"], cfg["reporting_seed_offset"] + cfg["reporting_seeds"])
    for level, slip in cfg["slip_levels"].items():
        for algo in list(ALGOS) + ["random"]:
            alpha = chosen.get(f"{level}|{algo}", 0.0)
            for s in seeds:
                tasks.append((algo, slip, alpha, s, cfg)); meta.append(level)
    res = _map(tasks, jobs)
    with open(os.path.join(out, "per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "slip", "algo", "alpha", "seed"] + SCORE_KEYS)
        for lv, r in zip(meta, res):
            w.writerow([lv, r["slip"], r["algo"], r["alpha"], r["seed"]] + [r["scores"][k] for k in SCORE_KEYS])
    for r, lv in zip(res, meta):
        r["level"] = lv
    with gzip.open(os.path.join(out, "episodes.csv.gz"), "wt", newline="") as f:   # log every episode
        w = csv.writer(f)
        w.writerow(["level", "slip", "algo", "alpha", "seed", "episode", "end_step", "ret", "length", "falls"])
        for r in res:
            for i in range(len(r["ep_returns"])):
                w.writerow([r["level"], r["slip"], r["algo"], r["alpha"], r["seed"], i + 1,
                            int(r["ep_end_step"][i]), float(r["ep_returns"][i]), int(r["ep_lens"][i]),
                            int(r["ep_falls"][i])])
    pickle.dump(res, open(os.path.join(out, "raw_report.pkl"), "wb"))
    print(f"  wrote {len(res)} reporting runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phase", default="all", choices=["tune", "report", "all"])
    ap.add_argument("--smoke", action="store_true", help="2 seeds, small budget, 2 alphas")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.smoke:
        cfg.update(train_steps=2000, eval_every=500, eval_episodes=5, alpha_grid=[0.1, 0.5],
                   tuning_seeds=2, reporting_seeds=2)
        if a.out == "results":
            a.out = "results_smoke"
    jobs = a.jobs if a.jobs is not None else cfg.get("n_jobs", 1)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**cfg, "env_id_gymnasium": ENV_ID, "python": sys.version, "gymnasium": gymnasium.__version__,
               "numpy": np.__version__, "packages": _versions(), "platform": platform.platform()},
              open(os.path.join(a.out, "run_config.json"), "w"), indent=2)
    refs = {lv: references(sl, cfg["epsilon"], cfg["gamma"]) for lv, sl in cfg["slip_levels"].items()}
    json.dump(refs, open(os.path.join(a.out, "references.json"), "w"), indent=2)
    for lv, r in refs.items():
        print(f"  DP {lv:6s} slip={r['slip']}: V*={r['v_star']:.2f}  eps-soft opt={r['v_eps_soft_opt']:.2f}  "
              f"Type1 margin={r['type1_margin']:.2f}")
    chosen = None
    if a.phase in ("tune", "all"):
        print("Tuning ..."); chosen = tune(cfg, a.out, jobs)
    if a.phase in ("report", "all"):
        if chosen is None:
            chosen = json.load(open(os.path.join(a.out, "chosen_alpha.json")))
        print("Reporting ..."); report(cfg, a.out, chosen, jobs)


if __name__ == "__main__":
    main()

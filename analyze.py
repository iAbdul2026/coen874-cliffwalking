"""Statistics, tables and figures from results/ (one score per seed is the unit of analysis).

  python analyze.py                               # everything, from results/
  python analyze.py --list                        # names of all tables and figures
  python analyze.py --only fig5 table_primary     # one command per figure or table
  python analyze.py --results results_smoke --boot 500     # smoke-test analysis

Protocol (examiner's brief, section 4): 95% percentile-bootstrap CIs; omnibus test across the three
algorithms (Welch's ANOVA, with Kruskal-Wallis as a robustness check); pairwise Welch's t-test (primary)
and Mann-Whitney U (robustness); Holm-Bonferroni correction; Hedges' g and the rank-biserial
correlation as effect sizes. Primary outcomes: online return and safety (cliff falls per 1,000 training
steps). Final performance = mean of the last K greedy-evaluation checkpoints (exact capped value).
Every item draws from its own random stream (seed + item name), so running one item alone gives
the same numbers as running all.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from agents import ALGOS
from cliff_mdp import COLS, GOAL, N_STATES, ROWS, START
from stats_utils import (boot_ci, boot_diff_ci, boot_diff_draws, hedges_g, holm, rank_biserial,
                         safe_kruskal, safe_mwu, welch_anova)

LEVELS = ["Low", "Medium", "High"]
NAMES = {"qlearning": "Q-learning", "sarsa": "SARSA", "expected_sarsa": "Expected SARSA", "random": "Random"}
COL = {"qlearning": "tab:orange", "sarsa": "tab:blue", "expected_sarsa": "tab:green", "random": "grey"}
MARK = {"qlearning": "s", "sarsa": "o", "expected_sarsa": "^", "random": "x"}
PAIRS = [("expected_sarsa", "sarsa"), ("expected_sarsa", "qlearning"), ("qlearning", "sarsa")]
W1, W2 = 3.45, 7.16          # IEEE column and full-page widths in inches

# name -> (label, higher_is_better, role)
METRICS = {
    "online_return": ("Online return (mean per training episode)", True, "primary"),
    "fall_rate_per_1k": ("Safety: cliff falls per 1,000 training steps", False, "primary"),
    "final_perf": ("Final greedy performance (last K checkpoints)", True, "secondary"),
    "gap_vs_vstar": ("Gap to DP optimum V* (final performance)", False, "secondary"),
    "auc_eval": ("Area under greedy-evaluation curve", True, "secondary"),
    "mean_episode_length": ("Mean training-episode length (steps)", False, "descriptive"),
    "falls_per_episode": ("Cliff falls per training episode", False, "descriptive"),
    "frac_episodes_with_fall": ("Share of training episodes with a fall", False, "descriptive"),
}
PRIMARY = [m for m, v in METRICS.items() if v[2] == "primary"]
TESTED = [m for m, v in METRICS.items() if v[2] in ("primary", "secondary")]

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "lines.linewidth": 1.0,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})


# ------------------------------------------------------------------------------ data
def load(results, last_k):
    df = pd.read_csv(os.path.join(results, "per_seed.csv"))
    raw = pickle.load(open(os.path.join(results, "raw_report.pkl"), "rb"))
    refs = json.load(open(os.path.join(results, "references.json")))
    by = {(r["level"], r["algo"], r["seed"]): r for r in raw}
    rows = [by[(l, a, s)] for l, a, s in zip(df.level, df.algo, df.seed)]
    df["final_perf"] = [float(np.mean(r["eval_exact"][-last_k:])) for r in rows]
    df["gap_vs_vstar"] = [refs[l]["v_star"] - v for l, v in zip(df.level, df.final_perf)]
    df["mean_episode_length"] = [float(np.mean(r["ep_lens"])) for r in rows]
    df["falls_per_episode"] = [float(np.mean(r["ep_falls"])) for r in rows]
    df["frac_episodes_with_fall"] = [float(np.mean(r["ep_falls"] > 0)) for r in rows]
    df["censored"] = df["censored"].astype(bool)      # steps_to_threshold stays NaN when never reached
    return df, raw, refs


class Ctx:
    def __init__(self, a, cfg):
        an = cfg.get("analysis", {})
        self.results = a.results
        self.K = a.last_k if a.last_k is not None else an.get("last_k_checkpoints", 5)
        self.B = a.boot if a.boot is not None else an.get("bootstrap_draws", 10000)
        self.seed = a.seed if a.seed is not None else an.get("analysis_seed", 12345)
        self.holm_family = a.holm_family or an.get("holm_family", "per_metric")
        self.alpha = an.get("significance_level", 0.05)
        self.tdir = os.path.join(a.results, "tables")
        self.fdir = os.path.join(a.results, "figures")
        os.makedirs(self.tdir, exist_ok=True); os.makedirs(self.fdir, exist_ok=True)
        self.df, self.raw, self.refs = load(a.results, self.K)
        self.run_cfg = json.load(open(os.path.join(a.results, "run_config.json")))
        self._summ = None

    def rng(self, name):
        return np.random.default_rng([self.seed] + [ord(c) for c in name])

    def vals(self, m, lv, al):
        return self.df[(self.df.level == lv) & (self.df.algo == al)][m].values

    def summary(self):
        if self._summ is None:
            rng = self.rng("summary"); rows = []
            for m in METRICS:
                for lv in LEVELS:
                    for al in list(ALGOS) + ["random"]:
                        x = self.vals(m, lv, al)
                        lo, hi = boot_ci(x, np.mean, self.B, rng); mlo, mhi = boot_ci(x, np.median, self.B, rng)
                        q1, q3 = np.percentile(x, [25, 75])
                        rows.append(dict(metric=m, role=METRICS[m][2], level=lv, algo=al, n=len(x), mean=x.mean(),
                                         mean_lo=lo, mean_hi=hi, median=np.median(x), median_lo=mlo, median_hi=mhi,
                                         iqr=q3 - q1, sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                                         minimum=x.min(), maximum=x.max()))
            self._summ = pd.DataFrame(rows)
        return self._summ


def _holm_apply(rows, pkey, outkey, groupkey):
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(groupkey(r), []).append(i)
    for idx in groups.values():
        for i, adj in zip(idx, holm([rows[i][pkey] for i in idx])):
            rows[i][outkey] = float(adj)


def run_tests(c):
    rng = c.rng("tests"); pair_rows, omni_rows = [], []
    for m in TESTED:
        for lv in LEVELS:
            groups = [c.vals(m, lv, a) for a in ALGOS]
            F, d1, d2, pw = welch_anova(groups); H, pk = safe_kruskal(groups)
            omni_rows.append(dict(metric=m, role=METRICS[m][2], level=lv, welch_anova_F=F, df1=d1, df2=d2,
                                  welch_anova_p=pw, kruskal_H=H, kruskal_p=pk))
            for A, Bn in PAIRS:
                a, b = c.vals(m, lv, A), c.vals(m, lv, Bn)
                with np.errstate(all="ignore"):
                    w = stats.ttest_ind(a, b, equal_var=False)
                _, mp = safe_mwu(a, b)
                lo, hi = boot_diff_ci(a, b, c.B, rng)
                pair_rows.append(dict(metric=m, role=METRICS[m][2], level=lv, A=A, B=Bn, mean_A=a.mean(), mean_B=b.mean(),
                                      diff_A_minus_B=a.mean() - b.mean(), diff_lo=lo, diff_hi=hi, hedges_g=hedges_g(a, b),
                                      welch_t=float(w.statistic), welch_p=float(w.pvalue), mwu_p=mp,
                                      rank_biserial=rank_biserial(a, b), higher_is_better=METRICS[m][1]))
    pooled = c.holm_family == "primary_pooled"
    gk = lambda r: "primary" if (pooled and r["role"] == "primary") else r["metric"]
    _holm_apply(pair_rows, "welch_p", "welch_p_holm", gk)
    _holm_apply(pair_rows, "mwu_p", "mwu_p_holm", gk)
    _holm_apply(omni_rows, "welch_anova_p", "welch_anova_p_holm", lambda r: r["metric"])
    _holm_apply(omni_rows, "kruskal_p", "kruskal_p_holm", lambda r: r["metric"])
    om = {(r["metric"], r["level"]): r for r in omni_rows}
    for r in pair_rows:
        r["sig_welch"] = bool(r["welch_p_holm"] < c.alpha); r["sig_mwu"] = bool(r["mwu_p_holm"] < c.alpha)
        o = om[(r["metric"], r["level"])]
        r["omnibus_welch_p_holm"], r["omnibus_kruskal_p_holm"] = o["welch_anova_p_holm"], o["kruskal_p_holm"]
        r["omnibus_significant"] = bool(o["welch_anova_p_holm"] < c.alpha)
    return pd.DataFrame(pair_rows), pd.DataFrame(omni_rows)


# ------------------------------------------------------------------------------ tables
def t_summary(c):
    c.summary().to_csv(os.path.join(c.tdir, "summary_by_group.csv"), index=False)


def t_tests(c):
    pairs, omni = run_tests(c)
    pairs.to_csv(os.path.join(c.tdir, "statistical_tests_pairwise.csv"), index=False)
    omni.to_csv(os.path.join(c.tdir, "statistical_tests_omnibus.csv"), index=False)


def t_contrasts(c):
    """Safety and online-return gaps between algorithms at each slip level, and how the gap changes with slip."""
    rng = c.rng("contrasts"); rows = []
    for m in PRIMARY:
        for A, Bn in PAIRS:
            d, pt = {}, {}
            for lv in LEVELS:
                a, b = c.vals(m, lv, A), c.vals(m, lv, Bn)
                d[lv] = boot_diff_draws(a, b, c.B, rng); pt[lv] = a.mean() - b.mean()
            items = [(lv, pt[lv], d[lv]) for lv in LEVELS]
            items += [("Medium minus Low", pt["Medium"] - pt["Low"], d["Medium"] - d["Low"]),
                      ("High minus Low", pt["High"] - pt["Low"], d["High"] - d["Low"])]
            for name, est, dr in items:
                rows.append(dict(metric=m, comparison=f"{NAMES[A]} minus {NAMES[Bn]}", quantity=name, estimate=est,
                                 lo=float(np.quantile(dr, 0.025)), hi=float(np.quantile(dr, 0.975))))
    pd.DataFrame(rows).to_csv(os.path.join(c.tdir, "gap_contrasts_primary.csv"), index=False)


def t_censoring(c):
    rows = []
    for lv in LEVELS:
        for al in ALGOS:
            g = c.df[(c.df.level == lv) & (c.df.algo == al)]
            reached = g.loc[~g.censored, "steps_to_threshold"]
            rows.append(dict(level=lv, algo=al, n=len(g), n_not_reached=int(g.censored.sum()),
                             share_not_reached=float(g.censored.mean()),
                             median_steps_among_reached=float(reached.median()) if len(reached) else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(c.tdir, "threshold_censoring.csv"), index=False)


def _fmt(r, nd=2):
    return f"{r['mean']:.{nd}f} [{r['mean_lo']:.{nd}f}, {r['mean_hi']:.{nd}f}]"


def t_main(c):
    s = c.summary(); rows = []
    for m in [k for k in METRICS if METRICS[k][2] != "descriptive"] + ["mean_episode_length"]:
        for lv in LEVELS:
            row = {"metric": m, "level": lv}
            for al in list(ALGOS) + ["random"]:
                row[NAMES[al]] = _fmt(s[(s.metric == m) & (s.level == lv) & (s.algo == al)].iloc[0])
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(c.tdir, "table_main_results.csv"), index=False)


def t_primary(c):
    pairs, omni = run_tests(c)
    p = pairs[pairs.role == "primary"].copy()
    p["comparison"] = [f"{NAMES[a]} minus {NAMES[b]}" for a, b in zip(p.A, p.B)]
    p["difference [95% CI]"] = [f"{d:.2f} [{lo:.2f}, {hi:.2f}]" for d, lo, hi in zip(p.diff_A_minus_B, p.diff_lo, p.diff_hi)]
    cols = ["metric", "level", "comparison", "difference [95% CI]", "hedges_g", "welch_p_holm", "mwu_p_holm",
            "rank_biserial", "omnibus_welch_p_holm", "omnibus_kruskal_p_holm"]
    p[cols].to_csv(os.path.join(c.tdir, "table_statistical_tests_primary.csv"), index=False)


def t_hyper(c):
    cfg, rows = c.run_cfg, []
    for k in ["epsilon", "gamma", "q_init", "train_steps", "episode_cap", "eval_every", "eval_episodes", "alpha_grid",
              "tuning_seeds", "tuning_seed_offset", "reporting_seeds", "reporting_seed_offset", "threshold_fraction"]:
        rows.append({"setting": k, "value": json.dumps(cfg[k])})
    for k, v in json.load(open(os.path.join(c.results, "chosen_alpha.json"))).items():
        rows.append({"setting": f"chosen alpha ({k})", "value": v})
    rows.append({"setting": "last_k_checkpoints (final performance)", "value": c.K})
    pd.DataFrame(rows).to_csv(os.path.join(c.tdir, "table_hyperparameters.csv"), index=False)
    pd.read_csv(os.path.join(c.results, "tuning.csv")).to_csv(os.path.join(c.tdir, "hyperparameter_tuning_grid.csv"), index=False)


def t_mdp(c):
    cfg = c.run_cfg
    rows = [("Environment", f"{cfg.get('env_id', 'CliffWalking-v1')}: 4 x 12 grid, custom slip wrapper (cliff_mdp.py)"),
            ("States S", "48 (state = row * 12 + col); start 36, goal 47, cliff 37 to 46"),
            ("Actions A", "4: 0 up, 1 right, 2 down, 3 left"),
            ("Transition P", "intended move with probability 1 - p; each perpendicular move with probability p/2; "
                             "never the opposite move; moves into a wall leave the state unchanged"),
            ("Slip levels p", ", ".join(f"{k} = {v}" for k, v in cfg["slip_levels"].items())),
            ("Reward R", "-1 per step; -100 on entering a cliff cell, after which the agent returns to the start"),
            ("Termination", "reaching the goal; a fall does not end the episode"),
            ("Truncation", f"{cfg['episode_cap']} steps per episode (the TD target still bootstraps)"),
            ("Discount gamma", str(cfg["gamma"]))]
    for lv in LEVELS:
        r = c.refs[lv]
        rows.append((f"DP optimum V* ({lv}, p = {r['slip']})", f"{r['v_star']:.2f}"))
    pd.DataFrame(rows, columns=["item", "specification"]).to_csv(os.path.join(c.tdir, "table_mdp.csv"), index=False)


def t_dp(c):
    pd.DataFrame(c.refs).T.to_csv(os.path.join(c.tdir, "dp_references.csv"))


# ------------------------------------------------------------------------------ figures
def _save(fig, c, name):
    fig.savefig(os.path.join(c.fdir, name)); plt.close(fig)


def f_learning(c):
    rng = c.rng("fig1"); raw, refs = c.raw, c.refs
    fig, ax = plt.subplots(2, 3, figsize=(W2, 4.2), sharex=True)
    for j, lv in enumerate(LEVELS):
        for al in ALGOS:
            runs = [r for r in raw if r["level"] == lv and r["algo"] == al]
            steps = runs[0]["eval_steps"]; M = np.array([r["eval_exact"] for r in runs]); O = []
            for r in runs:
                O.append([r["ep_returns"][(r["ep_end_step"] > s0 - 1000) & (r["ep_end_step"] <= s0)].mean()
                          if ((r["ep_end_step"] > s0 - 1000) & (r["ep_end_step"] <= s0)).any() else np.nan for s0 in steps])
            O = np.array(O)
            for i, Z in enumerate([M, O]):
                mean = np.nanmean(Z, 0); lo = np.full(len(steps), np.nan); hi = np.full(len(steps), np.nan)
                for t in range(len(steps)):
                    col = Z[:, t][~np.isnan(Z[:, t])]
                    if len(col) > 1:
                        lo[t], hi[t] = boot_ci(col, np.mean, min(c.B, 2000), rng)
                ax[i, j].plot(steps, mean, color=COL[al], label=NAMES[al])
                ax[i, j].fill_between(steps, lo, hi, color=COL[al], alpha=0.2)
        ax[0, j].axhline(refs[lv]["v_star"], color="k", ls=":", lw=0.8, label="DP optimum V*")
        ax[1, j].axhline(refs[lv]["v_eps_soft_opt"], color="k", ls=":", lw=0.8, label="Best epsilon-soft policy")
        ax[0, j].set_ylim(refs[lv]["v_star"] - 120 * (1 + j), refs[lv]["v_star"] + 10)
        ax[1, j].set_ylim(refs[lv]["v_eps_soft_opt"] - 250 * (1 + j), refs[lv]["v_eps_soft_opt"] + 20)
        ax[0, j].set_title(f"{lv} slip (p = {refs[lv]['slip']})"); ax[1, j].set_xlabel("Training steps")
    ax[0, 0].set_ylabel("Greedy-policy value"); ax[1, 0].set_ylabel("Online return per episode")
    hs, ls = ax[0, 0].get_legend_handles_labels(); h2, l2 = ax[1, 0].get_legend_handles_labels()
    hs += [h for h, l in zip(h2, l2) if l not in ls]; ls += [l for l in l2 if l not in ls]
    fig.legend(hs, ls, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    _save(fig, c, "fig1_learning_curves.pdf")


def f_distribution(c):
    fig, ax = plt.subplots(2, 3, figsize=(W2, 4.0))
    for i, m in enumerate(["final_perf", "fall_rate_per_1k"]):
        for j, lv in enumerate(LEVELS):
            data = [c.vals(m, lv, a) for a in ALGOS]; allv = np.concatenate(data)
            lo, hi = np.percentile(allv, 2), np.percentile(allv, 98); pad = 0.12 * max(hi - lo, 1e-9)
            lo, hi = lo - pad, hi + pad
            ax[i, j].boxplot(data, showfliers=False, tick_labels=[NAMES[a] for a in ALGOS], widths=0.5)
            jit = np.random.default_rng(0)
            for k, d in enumerate(data):
                x = jit.normal(k + 1, 0.05, len(d)); inside = (d >= lo) & (d <= hi)
                ax[i, j].scatter(x[inside], d[inside], s=6, alpha=0.7, color=COL[ALGOS[k]], zorder=3)
                for sel, mk, pos in ((d < lo, "v", lo), (d > hi, "^", hi)):     # no seed is dropped: clipped ones sit on the edge
                    if sel.any():
                        ax[i, j].scatter(x[sel], np.full(sel.sum(), pos), s=14, marker=mk, color=COL[ALGOS[k]], edgecolor="k", lw=0.5)
                        ax[i, j].annotate(f"{int(sel.sum())} beyond axis ({'min' if mk == 'v' else 'max'} {d[sel].min() if mk == 'v' else d[sel].max():.0f})",
                                          (k + 1, pos), xytext=(0, 6 if mk == "^" else -6), textcoords="offset points",
                                          ha=("left", "center", "right")[k], va="bottom" if mk == "^" else "top", fontsize=5)
            below = any((d < lo).any() for d in data); above = any((d > hi).any() for d in data)
            ax[i, j].set_ylim(lo - (0.22 if below else 0.08) * (hi - lo), hi + (0.22 if above else 0.08) * (hi - lo))
            ax[i, j].tick_params(axis="x", labelsize=5)
            ax[0, j].set_title(f"{lv} slip")
        ax[i, 0].set_ylabel("Final greedy performance" if m == "final_perf" else "Cliff falls per 1,000 steps")
    _save(fig, c, "fig2_distribution_per_seed.pdf")


def f_slip(c):
    s = c.summary(); fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3)); x = [c.refs[l]["slip"] for l in LEVELS]
    for k, m in enumerate(["online_return", "fall_rate_per_1k", "mean_episode_length"]):
        for al in ALGOS:
            g = s[(s.metric == m) & (s.algo == al)].set_index("level").loc[LEVELS]
            ax[k].errorbar(x, g["mean"], yerr=[g["mean"] - g["mean_lo"], g["mean_hi"] - g["mean"]], marker=MARK[al], ms=3,
                           capsize=2, color=COL[al], label=NAMES[al])
        ax[k].set_xlabel("Slip probability p"); ax[k].set_xticks(x)
        ax[k].set_ylabel({"online_return": "Online return", "fall_rate_per_1k": "Falls per 1,000 steps",
                          "mean_episode_length": "Mean episode length"}[m])
    ax[0].legend(); fig.subplots_adjust(wspace=0.5); _save(fig, c, "fig3_sensitivity_to_slip.pdf")


def f_policies(c):
    arrows = "^>v<"; fig, ax = plt.subplots(3, 3, figsize=(W2, 3.3))
    for i, lv in enumerate(LEVELS):
        for j, al in enumerate(ALGOS):
            Q = np.array([r["q_final"] for r in c.raw if r["level"] == lv and r["algo"] == al]); a_ = ax[i, j]
            a_.set_xlim(0, COLS); a_.set_ylim(ROWS, 0); a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
            for s in range(N_STATES):
                r, cc = divmod(s, COLS)
                if 37 <= s <= 46:
                    a_.add_patch(plt.Rectangle((cc, r), 1, 1, color="lightgrey")); continue
                if s == GOAL:
                    a_.text(cc + .5, r + .5, "G", ha="center", va="center", fontsize=6, color="green"); continue
                seen = Q[:, s, :].any(1)
                if seen.sum() == 0:
                    continue
                votes = np.bincount(Q[seen, s, :].argmax(1), minlength=4)
                a_.text(cc + .5, r + .5, arrows[votes.argmax()], ha="center", va="center", fontsize=6,
                        alpha=0.35 + 0.65 * votes.max() / seen.sum())
            a_.text(START % COLS + .5, START // COLS + .5, "S", ha="center", va="center", fontsize=6, color="red")
            a_.set_title(f"{lv} slip, {NAMES[al]}", fontsize=6)
    _save(fig, c, "fig4_policy_maps.pdf")


def f_tradeoff(c):
    """Safety versus online return: one point per algorithm and slip level, with 95% bootstrap CIs on both axes."""
    s = c.summary(); fig, ax = plt.subplots(figsize=(W1, 3.0))
    for al in ALGOS:
        xs, ys = [], []
        for lv in LEVELS:
            f = s[(s.metric == "fall_rate_per_1k") & (s.algo == al) & (s.level == lv)].iloc[0]
            o = s[(s.metric == "online_return") & (s.algo == al) & (s.level == lv)].iloc[0]
            ax.errorbar(f["mean"], o["mean"], xerr=[[f["mean"] - f["mean_lo"]], [f["mean_hi"] - f["mean"]]],
                        yerr=[[o["mean"] - o["mean_lo"]], [o["mean_hi"] - o["mean"]]], marker=MARK[al], ms=4, capsize=1.5,
                        color=COL[al], lw=0.6)
            ax.annotate(lv[0], (f["mean"], o["mean"]), xytext=(3, 3), textcoords="offset points", fontsize=5)
            xs.append(f["mean"]); ys.append(o["mean"])
        ax.plot(xs, ys, color=COL[al], lw=0.6, ls="--", label=NAMES[al])
    ax.set_xlabel("Cliff falls per 1,000 training steps (lower is safer)"); ax.set_ylabel("Online return per episode")
    ax.legend(); _save(fig, c, "fig5_safety_vs_online_return.pdf")


def f_alpha(c):
    """Learning-rate sensitivity on the tuning seeds (separate from the reporting seeds)."""
    path = os.path.join(c.results, "tuning_per_seed.csv")
    t = pd.read_csv(path); rng = c.rng("fig6"); fig, ax = plt.subplots(2, 3, figsize=(W2, 3.6), sharex=True)
    chosen = json.load(open(os.path.join(c.results, "chosen_alpha.json")))
    for j, lv in enumerate(LEVELS):
        for i, m in enumerate(["online_return", "fall_rate_per_1k"]):
            for al in ALGOS:
                g = t[(t.level == lv) & (t.algo == al)]; alphas = sorted(g.alpha.unique()); mu, lo, hi = [], [], []
                for a_ in alphas:
                    v = g[g.alpha == a_][m].values; mu.append(v.mean())
                    l_, h_ = boot_ci(v, np.mean, min(c.B, 2000), rng) if len(v) > 1 else (np.nan, np.nan); lo.append(l_); hi.append(h_)
                ax[i, j].plot(alphas, mu, marker=MARK[al], ms=3, color=COL[al], label=NAMES[al])
                ax[i, j].fill_between(alphas, lo, hi, color=COL[al], alpha=0.15)
                ca = chosen[f"{lv}|{al}"]
                if m == "online_return" and ca in alphas:
                    ax[i, j].plot([ca], [mu[alphas.index(ca)]], marker="*", ms=7, color=COL[al], mec="k", mew=0.4, ls="none")
            ax[0, j].set_title(f"{lv} slip")
        ax[1, j].set_xlabel("Learning rate alpha")
    ax[0, 0].set_ylabel("Online return"); ax[1, 0].set_ylabel("Falls per 1,000 steps"); ax[0, 0].legend()
    _save(fig, c, "fig6_sensitivity_to_alpha.pdf")


TABLES = {"summary": t_summary, "tests": t_tests, "contrasts": t_contrasts, "censoring": t_censoring,
          "table_main": t_main, "table_primary": t_primary, "table_hyper": t_hyper, "table_mdp": t_mdp, "dp": t_dp}
FIGURES = {"fig1": f_learning, "fig2": f_distribution, "fig3": f_slip, "fig4": f_policies, "fig5": f_tradeoff, "fig6": f_alpha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--config", default="config/default.json")
    ap.add_argument("--last-k", type=int, default=None); ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--holm-family", choices=["per_metric", "primary_pooled"], default=None)
    ap.add_argument("--only", nargs="+", default=None, help="names from --list")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    items = {**TABLES, **FIGURES}
    if a.list:
        print("tables :", ", ".join(TABLES)); print("figures:", ", ".join(FIGURES)); return
    bad = [n for n in (a.only or []) if n not in items]
    if bad:
        raise SystemExit(f"unknown item(s) {bad}; use --list")
    cfg = json.load(open(a.config)); c = Ctx(a, cfg)
    for name in (a.only or list(items)):
        items[name](c); print(f"  done: {name}")
    print(f"outputs in {c.tdir} and {c.fdir}; K={c.K}, B={c.B}, Holm family={c.holm_family}")


if __name__ == "__main__":
    main()

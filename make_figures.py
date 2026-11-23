"""Makes the figures used in the paper from results/*.csv.  Usage: python make_figures.py"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_experiments import ci95

R = "results"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": .3})

# ---- Fig A: NDCG@5 per epoch (learning curve) ------------------------------------------
df = pd.read_csv(f"{R}/raw_metrics.csv")
keep = ["Two-sided preference", "Supervised ranker", "Tuned claims-only score",
        "- Profile updates", "Full framework"]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for s, m in zip(keep, ["s", "^", "v", "x", "D"]):
    d = df[df.system == s].groupby("epoch").NDCG
    mu, ci = d.mean(), d.apply(lambda x: ci95(x))
    ax.errorbar(mu.index, mu.values, yerr=ci.values, marker=m, ms=3, lw=1, capsize=1.5, label=s)
ax.set_xlabel("Epoch (0 = cold start)"); ax.set_ylabel("NDCG@5")
ax.set_ylim(0.68, 0.94); ax.legend(fontsize=5.5, loc="lower left"); fig.tight_layout()
fig.savefig(f"{R}/fig_learning_curve.png", dpi=300); plt.close(fig)

# ---- Fig B: main comparison bar chart ----------------------------------------------------
d = df[df.epoch.between(1, 5)]
ps = d.groupby(["system", "seed"]).NDCG.mean().reset_index()
order = ["TF-IDF keyword", "Semantic (skill graph)", "Supervised ranker", "Two-sided preference",
         "Tuned claims-only score", "- Verification", "- Profile updates", "- Bilateral preferences",
         "- Feasibility gate", "Full framework"]
mu = ps.groupby("system").NDCG.mean().reindex(order)
ci = ps.groupby("system").NDCG.apply(ci95).reindex(order)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.barh(range(len(order)), mu.values, xerr=ci.values, color="#4477aa", capsize=1.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6); ax.invert_yaxis()
ax.set_xlabel("NDCG@5 (mean ± 95% CI)"); ax.set_xlim(0, 1)
fig.tight_layout(); fig.savefig(f"{R}/fig_comparison.png", dpi=300); plt.close(fig)

# ---- Fig C: sensitivity to claim-inflation rate -----------------------------------------
sw = pd.read_csv(f"{R}/sweeps_raw.csv")
inf = sw[(sw.sweep == "inflator_rate") & sw.epoch.between(1, 5)]
fig, ax = plt.subplots(figsize=(3.5, 2.3))
for s, m in [("Two-sided preference", "s"), ("Supervised ranker", "^"), ("Tuned claims-only score", "v"),
             ("Full framework", "D")]:
    g = inf[inf.system == s].groupby(["value", "seed"]).NDCG.mean().reset_index().groupby("value").NDCG
    ax.errorbar(g.mean().index, g.mean().values, yerr=g.apply(ci95).values, marker=m, ms=3, lw=1,
                capsize=1.5, label=s)
ax.set_xlabel("Share of workers who inflate claims"); ax.set_ylabel("NDCG@5"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{R}/fig_inflation_sweep.png", dpi=300); plt.close(fig)
print("figures written")

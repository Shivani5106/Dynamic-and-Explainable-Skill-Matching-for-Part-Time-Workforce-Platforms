"""Builds results/PAPER_RESULTS.md (tables + paste-ready text) from the CSVs, so every number is traceable."""
import json
import numpy as np, pandas as pd
from run_experiments import ci95
R = "results"
raw = pd.read_csv(f"{R}/raw_metrics.csv"); main = raw[raw.epoch.between(1, 5)]
ps = main.groupby(["system", "seed"])[["P1", "Pk", "Rk", "F1", "NDCG"]].mean().reset_index()
def ms(sys_, m): x = ps[ps.system == sys_][m]; return x.mean(), ci95(x)
def f(sys_, m, d=3): mu, c = ms(sys_, m); return f"{mu:.{d}f} ± {c:.{d}f}"
cold = raw[raw.epoch == 0].groupby(["system", "seed"]).NDCG.mean().reset_index()
def cn(s): return cold[cold.system == s].NDCG.mean()
curve = pd.read_csv(f"{R}/learning_curve.csv", index_col=0)
tests = pd.read_csv(f"{R}/paired_tests.csv").set_index("comparison")
W = json.load(open(f"{R}/weights.json")); lat = json.load(open(f"{R}/latency.json"))
vq = pd.read_csv(f"{R}/verification_quality.csv").drop(columns="seed").mean()
fr = pd.read_csv(f"{R}/fairness_raw.csv"); fr = fr[fr.epoch.between(1, 5)]
fg = fr.groupby(["system", "seed"])[["share_B_top", "share_B_relevant", "rep_gap", "prec_A", "prec_B"]].mean().reset_index()
asg = pd.read_csv(f"{R}/assignment_raw.csv").groupby(["system", "seed"])[["success", "mean_gain"]].mean().reset_index()
sw = pd.read_csv(f"{R}/sweeps_raw.csv")

rows = [("TF-IDF keyword", "Keyword baseline (i)"), ("Semantic (skill graph)", "Semantic baseline (ii); skill graph stands in for SBERT"),
        ("Supervised ranker", "Gradient-boosted ranker (iii), trained on earlier epochs only"),
        ("Two-sided preference", "Two-sided preference baseline (iv)"),
        ("Tuned claims-only score", "Control: same weight tuning, but no verification and no performance"),
        ("Full framework", "Proposed framework"), ("Oracle (ceiling)", "Ground-truth ceiling, not a real system")]
t1 = "| System | P@1 | P@5 | R@5 | F1@5 | NDCG@5 | Note |\n|---|---|---|---|---|---|---|\n"
for s, n in rows:
    t1 += f"| {s} | {f(s,'P1')} | {f(s,'Pk')} | {f(s,'Rk')} | {f(s,'F1')} | {f(s,'NDCG')} | {n} |\n"

abl = [("Full framework", "All components"), ("- Verification", "Raw claims, no verified evidence, claim-based gate"),
       ("- Profile updates", "lambda = 0: no feedback, S_perf constant"), ("- Bilateral preferences", "delta = 0"),
       ("- Feasibility gate", "No hard constraints before ranking")]
t2 = "| Variant | NDCG@5 | Change vs full | Paired p (t-test) | Seeds where full is better | What was removed |\n|---|---|---|---|---|---|\n"
for s, n in abl:
    if s == "Full framework":
        t2 += f"| {s} | {f(s,'NDCG')} | – | – | – | {n} |\n"
    else:
        r = tests.loc[f"Full vs {s}"]
        t2 += f"| {s} | {f(s,'NDCG')} | {-r.mean_diff:+.3f} | {'<0.001' if r.t_p<0.001 else f'{r.t_p:.3f}'} | {int(r.seeds_full_better)}/{int(r.n)} | {n} |\n"

lc = "| System | " + " | ".join(f"Epoch {e}" for e in curve.columns) + " |\n|---|" + "---|" * len(curve.columns) + "\n"
for s in ["Full framework", "- Profile updates", "- Verification", "Two-sided preference", "Supervised ranker"]:
    lc += f"| {s} | " + " | ".join("–" if pd.isna(v) else f"{v:.3f}" for v in curve.loc[s]) + " |\n"

inf = sw[(sw.sweep == "inflator_rate") & sw.epoch.between(1, 5)].groupby(["system", "value", "seed"]).NDCG.mean().reset_index()
pv = inf.pivot_table(index=["value", "seed"], columns="system", values="NDCG")
t3 = "| Share of workers who inflate claims | Tuned claims-only | Two-sided preference | Supervised ranker | Full - Verification | Full framework | Gain of verification (Full minus -Verification) |\n|---|---|---|---|---|---|---|\n"
for v in [0.0, 0.15, 0.35, 0.6]:
    g = pv.loc[v]; d = g["Full framework"] - g["- Verification"]
    t3 += (f"| {v:.2f} | {g['Tuned claims-only score'].mean():.3f} | {g['Two-sided preference'].mean():.3f} | "
           f"{g['Supervised ranker'].mean():.3f} | {g['- Verification'].mean():.3f} | {g['Full framework'].mean():.3f} | "
           f"{d.mean():+.3f} ± {ci95(d):.3f} |\n")

lam = {k: ps[ps.system == k].NDCG.mean() for k in ["Full (lam=0.1)", "Full framework", "Full (lam=0.5)", "Full (lam=0.9)"]}
t4 = "| System | Share of group B in top-5 | Share of group B among relevant workers | Representation gap | Precision@5 on group A slots | Precision@5 on group B slots |\n|---|---|---|---|---|---|\n"
for s in ["TF-IDF keyword", "Two-sided preference", "- Verification", "Full framework"]:
    g = fg[fg.system == s]
    t4 += (f"| {s} | {g.share_B_top.mean():.3f} | {g.share_B_relevant.mean():.3f} | {g.rep_gap.mean():+.3f} ± {ci95(g.rep_gap):.3f} | "
           f"{g.prec_A.mean():.3f} | {g.prec_B.mean():.3f} |\n")
ab = sw[(sw.sweep == "access_B_fair") & (sw.system == "Full framework") & sw.epoch.between(1, 5)].groupby(["value", "seed"]).rep_gap.mean().reset_index()
abn = sw[(sw.sweep == "access_B") & sw.epoch.between(1, 5)].groupby(["system", "value", "seed"]).NDCG.mean().reset_index()
t5 = "| Verification access of group B | Representation gap of group B (full framework) | NDCG@5 full | NDCG@5 without verification |\n|---|---|---|---|\n"
for v in [0.8, 0.5, 0.2]:
    g = ab[ab.value == v].rep_gap
    t5 += (f"| {v:.1f} | {g.mean():+.3f} ± {ci95(g):.3f} | {abn[(abn.system=='Full framework')&(abn.value==v)].NDCG.mean():.3f} | "
           f"{abn[(abn.system=='- Verification')&(abn.value==v)].NDCG.mean():.3f} |\n")

t6 = "| System | Assignment success rate | Mean true quality of assigned pairs |\n|---|---|---|\n"
for s in ["TF-IDF keyword", "Semantic (skill graph)", "Supervised ranker", "Two-sided preference", "Full framework"]:
    g = asg[asg.system == s]
    t6 += f"| {s} | {g.success.mean():.3f} ± {ci95(g.success):.3f} | {g.mean_gain.mean():.3f} ± {ci95(g.mean_gain):.3f} |\n"

wt = W["tuned"]["Full framework"]["weights"]
fw = ", ".join(f"{n} {v:.2f}" for n, v in zip(["skill", "ver", "exp", "pref", "avail", "perf"], wt))
d_ver, d_upd = tests.loc["Full vs - Verification"].mean_diff, tests.loc["Full vs - Profile updates"].mean_diff
d_ctl = tests.loc["Full vs Tuned claims-only score"].mean_diff

md = f"""# Results for the paper (generated by make_report.py from results/*.csv)

IMPORTANT: every number below comes from a SYNTHETIC simulation (600 workers, 8 skill clusters/48 skills,
100 fresh tasks per epoch, 6 epochs, 20 independent random seeds). It validates that the framework works end-to-end and
shows how its components behave under stated assumptions. It is NOT evidence about real platforms.

## Table A. Main comparison (epochs 1-5, mean ± 95% CI over 20 seeds)
{t1}
Metric definitions: P@k = share of top-k that are truly suitable; R@k = suitable workers found in top-k divided by min(k, number of suitable workers);
F1@5 = harmonic mean of P@5 and R@5; NDCG@5 uses graded true suitability. Evaluated on tasks with at least one suitable worker.

## Table B. Ablation (weights re-tuned for every variant on separate development seeds)
{t2}
## Table C. Learning curve, NDCG@5 by epoch (epoch 0 = cold start; each epoch uses only earlier epochs' outcomes)
{lc}
## Table D. Sensitivity to claim inflation (10 separate seeds; weights fixed from the default setting)
{t3}
## Table E. Fairness (group B has lower verification access by assumption: 0.50 vs 0.80)
{t4}
Group membership is never used as a model feature.

## Table F. Effect of the verification-access gap
{t5}
## Table G. Two-sided stable assignment (task-proposing deferred acceptance, 60 tasks per epoch, epochs 1-5)
{t6}
## Other numbers
- Learned weights (full framework, dev-tuned): {fw}.
- lambda sensitivity (NDCG@5, weights fixed): 0.1 -> {lam['Full (lam=0.1)']:.3f}; 0.3 -> {lam['Full framework']:.3f}; 0.5 -> {lam['Full (lam=0.5)']:.3f}; 0.9 -> {lam['Full (lam=0.9)']:.3f}.
- Weight robustness: 30 random ±30% perturbations of the weights -> NDCG@5 {ps[ps.system=='Full (weight perturbation)'].NDCG.mean():.3f} vs {ps[ps.system=='Full framework'].NDCG.mean():.3f}.
- Cold start (epoch 0) NDCG@5: full {cn('Full framework'):.3f}; tuned claims-only {cn('Tuned claims-only score'):.3f}; without verification {cn('- Verification'):.3f}; two-sided preference {cn('Two-sided preference'):.3f}.
- Verification module in simulation: precision of accepted claims {vq.precision:.3f}; recall of submitted true skills {vq.recall_submitted:.3f}; decision accuracy {vq.decision_accuracy:.3f};
  false claims rejected {vq.false_claims_rejected:.3f}, false claims wrongly accepted {vq.false_claims_verified:.3f}; evidence submitted for {vq.claims_submitted:.3f} of claims.
  (These are set by the simulated verifier's assumed true/false-accept rates 0.92/0.08 and submission rates, so they are a consistency check, not a finding.)
- Latency (single CPU core, 600 workers): {lat['inference_with_explanation_ms']:.1f} ms per recommendation including explanation; {lat['profile_update_ms']:.4f} ms per profile update.
- Explanation coverage: 100% by construction (every recommendation carries factor contributions and blocking reasons).
- Summary of increments: verification adds {d_ver:+.3f} NDCG@5 on top of everything else; profile updates add {d_upd:+.3f}; the full framework beats the tuned claims-only control by {d_ctl:+.3f}.
"""
open(f"{R}/PAPER_RESULTS.md", "w").write(md)
print(md)

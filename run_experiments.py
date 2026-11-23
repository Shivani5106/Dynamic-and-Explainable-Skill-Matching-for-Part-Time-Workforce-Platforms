"""Reproduces every number reported in the paper.  Usage:  python run_experiments.py [--quick]"""
import json, sys, time
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy import stats

from skillmatch.simulate import Params
from skillmatch.pipeline import simulate, rank_metrics, K
from skillmatch.model import Platform, COMP_NAMES, deferred_acceptance

QUICK = "--quick" in sys.argv
DEV_SEEDS = list(range(100, 105))
TEST_SEEDS = list(range(20 if not QUICK else 5))
SWEEP_SEEDS = list(range(200, 210 if not QUICK else 203))
LAM = 0.3                                   # fixed a priori, then swept
EPOCHS_MAIN = range(1, 6)                   # epochs 1-5 (all systems have history); epoch 0 = cold start
OUT = "results"

# variant -> (lam key, use_verification, use_gate, active weight indices)
VARIANTS = {
    "Full framework":          (LAM, True,  True,  [0, 1, 2, 3, 4, 5]),
    "- Verification":          (LAM, False, True,  [0, 2, 3, 4, 5]),
    "- Profile updates":       (0.0, True,  True,  [0, 1, 2, 3, 4]),
    "- Bilateral preferences": (LAM, True,  True,  [0, 1, 2, 4, 5]),
    "- Feasibility gate":      (LAM, True,  False, [0, 1, 2, 3, 4, 5]),
    # tuned-weight control: same weight search budget, but claims only (no verification, no performance)
    "Tuned claims-only score": (0.0, False, True,  [0, 2, 3, 4]),
}


def expand(w_active, active):
    w = np.zeros(6)
    w[active] = w_active
    return w / w.sum()


# ---------------------------------------------------------------- weight tuning on dev seeds
def tune_weights(params, n_samples=400):
    rng = np.random.default_rng(12345)
    data = []
    for sd in DEV_SEEDS:
        for ed in simulate(sd, params, lams=(0.0, LAM)):
            if ed.epoch in EPOCHS_MAIN:
                data.append(ed)
    cache = {}
    best = {}
    for name, (lam, use_ver, use_gate, active) in VARIANTS.items():
        samples = [np.ones(len(active)) / len(active)] + \
                  [rng.dirichlet(np.ones(len(active))) for _ in range(n_samples)] + \
                  [rng.dirichlet(0.3 * np.ones(len(active))) for _ in range(n_samples // 2)]
        scores = []
        for ws in samples:
            w = expand(ws, active)
            vals = []
            for i, ed in enumerate(data):
                c = ed.comps[lam]
                m = ed.platforms[lam].score(None, w, use_gate=use_gate, use_verification=use_ver, comps=c)
                r, _, _ = rank_metrics(m, ed.gt, np.random.default_rng(i))
                vals.append(r["NDCG"])
            scores.append(np.mean(vals))
        j = int(np.argmax(scores))
        best[name] = dict(weights=expand(samples[j], active).tolist(), dev_ndcg=float(scores[j]),
                          equal_weight_dev_ndcg=float(scores[0]))
        print(f"  tuned {name:24s} dev NDCG@5 = {scores[j]:.4f}  (equal weights {scores[0]:.4f})")
    return best


# ---------------------------------------------------------------- one evaluation run
def evaluate_seed(seed, params, weights, lams=(0.0, LAM), extras=False, perturb=None):
    rng = np.random.default_rng([seed, 99])
    rows, fair, match = [], [], []
    vq = None
    for ed in simulate(seed, params, lams=lams):
        gt, w = ed.gt, ed.world
        systems = {"Oracle (ceiling)": gt["gain"]}
        systems.update(ed.base)
        for name, (lam, use_ver, use_gate, _) in VARIANTS.items():
            if lam not in ed.comps:
                continue
            systems[name] = ed.platforms[lam].score(None, np.array(weights[name]), use_gate=use_gate,
                                                    use_verification=use_ver, comps=ed.comps[lam])
        tops = {}
        for name, sc in systems.items():
            r, top, valid = rank_metrics(sc, gt, rng)
            rows.append(dict(seed=seed, epoch=ed.epoch, system=name, **r))
            tops[name] = (top, valid)
        # extra lambda values (sensitivity): full framework with other lam, fixed weights
        for lam in lams:
            if lam in (0.0, LAM):
                continue
            sc = ed.platforms[lam].score(None, np.array(weights["Full framework"]), comps=ed.comps[lam])
            r, _, _ = rank_metrics(sc, gt, rng)
            rows.append(dict(seed=seed, epoch=ed.epoch, system=f"Full (lam={lam})", **r))
        if perturb is not None:
            sc_list = []
            for wv in perturb:
                sc = ed.platforms[LAM].score(None, wv, comps=ed.comps[LAM])
                r, _, _ = rank_metrics(sc, gt, rng)
                rows.append(dict(seed=seed, epoch=ed.epoch, system="Full (weight perturbation)", **r))
        if not extras:
            continue
        # --- fairness: representation + precision by group, over valid tasks
        rel = gt["relevant"]
        for name in ["TF-IDF keyword", "Two-sided preference", "- Verification", "Full framework"]:
            top, valid = tops[name]
            gB = w.group[top[valid]] == 1
            base_B = rel[valid][:, w.group == 1].sum() / rel[valid].sum()
            hit = np.take_along_axis(rel[valid], top[valid], axis=1)
            fair.append(dict(seed=seed, epoch=ed.epoch, system=name,
                             share_B_top=gB.mean(), share_B_relevant=base_B,
                             rep_gap=gB.mean() - base_B,
                             prec_A=hit[~gB].mean() if (~gB).any() else np.nan,
                             prec_B=hit[gB].mean() if gB.any() else np.nan))
        # --- two-sided stable assignment on a batch of 60 tasks
        if ed.epoch in EPOCHS_MAIN:
            n = 60
            pref_w = ed.comps[LAM]["pref"][:n] * (ed.comps[LAM]["avail"][:n] > 0)
            for name in ["TF-IDF keyword", "Semantic (skill graph)", "Two-sided preference",
                         "Supervised ranker", "Full framework"]:
                if name not in systems:
                    continue
                sc = systems[name][:n]
                assign = deferred_acceptance(sc, np.where(pref_w > 0, pref_w, -np.inf))
                if assign:
                    ok = [rel[t, wk] for t, wk in assign.items()]
                    g = [gt["gain"][t, wk] for t, wk in assign.items()]
                    match.append(dict(seed=seed, epoch=ed.epoch, system=name, success=np.mean(ok),
                                      mean_gain=np.mean(g), matched=len(assign) / n))
        if vq is None:     # verification quality at onboarding (does not change over epochs)
            sub = w.submit
            tp = (w.verified & w.has).sum(); fp = (w.verified & ~w.has).sum()
            vq = dict(seed=seed, precision=tp / max(tp + fp, 1),
                      recall_submitted=tp / max((sub & w.has).sum(), 1),
                      decision_accuracy=((w.verified & w.has) | (w.rejected & ~w.has)).sum() / max(sub.sum(), 1),
                      false_claims_rejected=(w.rejected & ~w.has).sum() / max((w.claimed & ~w.has).sum(), 1),
                      false_claims_verified=(w.verified & ~w.has).sum() / max((w.claimed & ~w.has).sum(), 1),
                      claims_submitted=sub.sum() / w.claimed.sum(),
                      inflators=w.inflator.mean(), false_claim_share=(w.claimed & ~w.has).sum() / w.claimed.sum())
    return rows, fair, match, vq


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def summarize(df, metrics=("P1", "Pk", "Rk", "F1", "NDCG"), epochs=EPOCHS_MAIN):
    d = df[df.epoch.isin(list(epochs))]
    per_seed = d.groupby(["system", "seed"])[list(metrics)].mean().reset_index()
    g = per_seed.groupby("system")
    out = pd.DataFrame({m: g[m].mean() for m in metrics})
    for m in metrics:
        out[m + "_ci"] = g[m].apply(ci95)
    return out, per_seed


def timing(params, weights):
    import time as _t
    from skillmatch.simulate import generate_world, generate_tasks
    rng = np.random.default_rng(7)
    w = generate_world(rng, params)
    tasks = generate_tasks(rng, 200, params)
    plat = Platform(w, LAM)
    wv = np.array(weights["Full framework"])
    plat.recommend(tasks, 0, wv)       # warm-up
    t0 = _t.perf_counter()
    for i in range(200):
        recs = plat.recommend(tasks, i, wv)
        plat.explain(tasks, i, recs[0][0], wv) if recs else None
    inf_ms = (_t.perf_counter() - t0) / 200 * 1000
    t0 = _t.perf_counter()
    for i in range(2000):
        plat.record_outcome(i % params.n_workers, int(tasks.cat[i % 200]), np.where(tasks.req[i % 200])[0], 0.7)
    upd_ms = (_t.perf_counter() - t0) / 2000 * 1000
    return dict(inference_with_explanation_ms=inf_ms, profile_update_ms=upd_ms,
                n_workers=params.n_workers)


def main():
    t_start = time.time()
    params = Params()
    print("== tuning weights on dev seeds", DEV_SEEDS)
    tuned = tune_weights(params, n_samples=400 if not QUICK else 100)
    weights = {k: v["weights"] for k, v in tuned.items()}
    json.dump(dict(lam=LAM, tuned=tuned, params=params.__dict__), open(f"{OUT}/weights.json", "w"), indent=1)

    print("== main test run, seeds", TEST_SEEDS[0], "-", TEST_SEEDS[-1])
    rng = np.random.default_rng(2024)
    base_w = np.array(weights["Full framework"])
    perturb = []
    for _ in range(30):
        v = base_w * np.exp(rng.normal(0, 0.3, 6))
        perturb.append(v / v.sum())
    rows, fair, match, vqs = [], [], [], []
    for sd in TEST_SEEDS:
        r, f, m, v = evaluate_seed(sd, params, weights, lams=(0.0, LAM, 0.1, 0.5, 0.9), extras=True, perturb=perturb)
        rows += r; fair += f; match += m; vqs.append(v)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/raw_metrics.csv", index=False)
    summ, per_seed = summarize(df)
    summ.to_csv(f"{OUT}/main_summary.csv")
    cold, _ = summarize(df, epochs=[0])
    cold.to_csv(f"{OUT}/cold_start_summary.csv")
    print(summ[["P1", "Pk", "Rk", "F1", "NDCG"]].round(3).to_string())

    # paired tests: full vs each other system (per-seed NDCG)
    piv = per_seed.pivot(index="seed", columns="system", values="NDCG")
    tests = []
    for other in piv.columns:
        if other in ("Full framework",) or other.startswith("Full (") or other.startswith("Oracle"):
            continue
        d = piv["Full framework"] - piv[other]
        t, p = stats.ttest_rel(piv["Full framework"], piv[other])
        try:
            wp = stats.wilcoxon(d).pvalue
        except ValueError:
            wp = np.nan
        tests.append(dict(comparison=f"Full vs {other}", mean_diff=d.mean(), ci=ci95(d), t_p=p, wilcoxon_p=wp,
                          seeds_full_better=int((d > 0).sum()), n=len(d)))
    pd.DataFrame(tests).to_csv(f"{OUT}/paired_tests.csv", index=False)

    pd.DataFrame(fair).to_csv(f"{OUT}/fairness_raw.csv", index=False)
    pd.DataFrame(match).to_csv(f"{OUT}/assignment_raw.csv", index=False)
    pd.DataFrame(vqs).to_csv(f"{OUT}/verification_quality.csv", index=False)
    json.dump(timing(params, weights), open(f"{OUT}/latency.json", "w"), indent=1)

    # learning curve (per epoch)
    curve = df[df.system.isin(["TF-IDF keyword", "Semantic (skill graph)", "Two-sided preference",
                               "Supervised ranker", "Full framework", "- Profile updates", "- Verification"])]
    curve.groupby(["system", "epoch"]).NDCG.mean().unstack().to_csv(f"{OUT}/learning_curve.csv")

    # sensitivity: inflation rate and verification-access gap
    print("== sweeps, seeds", SWEEP_SEEDS[0], "-", SWEEP_SEEDS[-1])
    sweep_rows = []
    for infl in [0.0, 0.15, 0.35, 0.6]:
        for sd in SWEEP_SEEDS:
            r, _, _, _ = evaluate_seed(sd, replace(params, inflator_rate=infl), weights, lams=(0.0, LAM))
            sweep_rows += [dict(sweep="inflator_rate", value=infl, **x) for x in r]
    for accB in [0.8, 0.5, 0.2]:
        for sd in SWEEP_SEEDS:
            r, f, _, _ = evaluate_seed(sd, replace(params, verif_access_b=accB), weights, lams=(0.0, LAM), extras=True)
            sweep_rows += [dict(sweep="access_B", value=accB, **x) for x in r]
            sweep_rows += [dict(sweep="access_B_fair", value=accB, **x) for x in f]
    pd.DataFrame(sweep_rows).to_csv(f"{OUT}/sweeps_raw.csv", index=False)
    print("done in", round(time.time() - t_start), "s")


if __name__ == "__main__":
    main()

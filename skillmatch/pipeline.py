"""Time-aware simulation: each epoch evaluates fresh tasks using only logs from earlier epochs."""
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from .taxonomy import SKILL_NAMES
from .simulate import Params, generate_world, generate_tasks, ground_truth
from .model import Platform, static_components

K = 5


class TfidfScorer:
    def __init__(self, world):
        wdocs = [" ".join(SKILL_NAMES[s] for s in np.where(r)[0]).lower() for r in world.claimed]
        self.vec = TfidfVectorizer()
        self.W = self.vec.fit_transform(wdocs)

    def score(self, tasks):
        tdocs = [" ".join(SKILL_NAMES[s] for s in np.where(r)[0]).lower() for r in tasks.req]
        return (self.vec.transform(tdocs) @ self.W.T).toarray()


@dataclass
class EpochData:
    epoch: int
    world: object
    tasks: object
    gt: dict
    comps: dict          # lam -> component dict (proposed system state at this epoch)
    base: dict           # baseline name -> score matrix [T,N]
    platforms: dict      # lam -> live Platform (for demos)


def simulate(seed, params: Params, lams=(0.0, 0.3)):
    """Generator: yields EpochData for epochs 0..n-1; the consumer processes it, then the
    generator produces that epoch's historical hires and updates the platform states."""
    world = generate_world(np.random.default_rng([seed, 1]), params)
    r_tasks = np.random.default_rng([seed, 2])
    r_logs = np.random.default_rng([seed, 3])
    tf = TfidfScorer(world)
    plats = {lam: Platform(world, lam) for lam in lams}
    X, y = [], []

    for e in range(params.n_epochs):
        tasks = generate_tasks(r_tasks, params.tasks_per_epoch, params)
        gt = ground_truth(world, tasks)
        stat = static_components(world, tasks)
        tfs = tf.score(tasks)
        feats = _sup_features(stat, tfs)
        base = {
            "TF-IDF keyword": tfs,
            "Semantic (skill graph)": stat["skill_raw"],
            "Two-sided preference": _two_sided(stat),
        }
        if len(y) >= 50:
            gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
            gbr.fit(np.array(X), np.array(y))
            T, N = tfs.shape
            pred = gbr.predict(feats.reshape(T * N, -1)).reshape(T, N)
            # logs contain only feasible hires, so the learned model cannot see feasibility;
            # apply the same hard availability/preference factors the other baselines use.
            base["Supervised ranker"] = pred * stat["avail"] * stat["pref"]
        yield EpochData(e, world, tasks, gt, {l: p.components(tasks) for l, p in plats.items()},
                        base, plats)

        # ---- historical hiring for this epoch (claim-based, as on a static platform) ----
        ltasks = generate_tasks(r_logs, params.logs_per_epoch, params)
        lgt = ground_truth(world, ltasks)
        lstat = static_components(world, ltasks)
        ltf = tf.score(ltasks)
        lfeat = _sup_features(lstat, ltf)
        feasible = lstat["gate_core"]
        for t in range(len(ltasks)):
            cand = np.where(feasible[t] & (lstat["pref"][t] >= 0.6))[0]
            if len(cand) == 0:
                continue
            top = cand[np.argsort(-ltf[t, cand])[:10]]
            wk = int(r_logs.choice(top))
            q = float(np.clip(lgt["comp"][t, wk] + r_logs.normal(0, params.outcome_noise), 0, 1))
            X.append(lfeat[t, wk])
            y.append(q)
            req_idx = np.where(ltasks.req[t])[0]
            for p in plats.values():
                p.record_outcome(wk, int(ltasks.cat[t]), req_idx, q)


def _sup_features(stat, tfs):
    return np.stack([stat["skill_raw"], stat["exp"], stat["pref"], stat["avail"], tfs], axis=-1)


def _two_sided(stat):
    worker_side = stat["pref"]
    task_side = 0.5 * stat["skill_raw"] + 0.5 * stat["exp"]
    mutual = 2 * worker_side * task_side / np.maximum(worker_side + task_side, 1e-9)
    return mutual * stat["avail"]


# ------------------------------------------------------------------------------------------
def rank_metrics(scores, gt, rng, k=K):
    """Per-task Precision@1/@k, capped Recall@k, NDCG@k, over tasks with >=1 relevant worker."""
    rel, gain = gt["relevant"], gt["gain"]
    valid = rel.sum(1) >= 1
    s = np.where(np.isfinite(scores), scores, -1e9) + rng.random(scores.shape) * 1e-9
    part = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.arange(len(s))[:, None]
    top = part[rows, np.argsort(-s[rows, part], axis=1)]
    hit = rel[rows, top]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gain[rows, top] * disc).sum(1)
    ideal = (-np.sort(-gain, axis=1)[:, :k] * disc).sum(1)
    nrel = np.minimum(rel.sum(1), k)
    out = dict(P1=hit[:, 0], Pk=hit.mean(1), Rk=hit.sum(1) / np.maximum(nrel, 1),
               NDCG=dcg / np.maximum(ideal, 1e-12))
    res = {m: float(v[valid].mean()) for m, v in out.items()}
    p, r = res["Pk"], res["Rk"]
    res["F1"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    res["n_tasks"] = int(valid.sum())
    return res, top, valid

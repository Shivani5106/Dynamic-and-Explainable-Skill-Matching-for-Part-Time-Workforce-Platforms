"""Proposed framework (paper Section III).

M(w,t) = a*S_skill + b*S_ver + c*S_exp + d*S_pref + e*S_avail + z*S_perf
with a feasibility gate (availability, zone, mandatory verified evidence) and the
profile update  p_new(s) = (1 - lam_eff) p_old(s) + lam_eff * E(s).
"""
import numpy as np
from .taxonomy import SIM, SKILL_NAMES, CLUSTER_NAMES, N_SKILLS, N_CLUSTERS
from .simulate import (World, Tasks, req_norm, avail_matrices)

COMP_NAMES = ["skill", "ver", "exp", "pref", "avail", "perf"]
FACTOR_LABELS = {"skill": "Semantic skill fit", "ver": "Verified capability",
                 "exp": "Experience", "pref": "Preference alignment",
                 "avail": "Availability", "perf": "Past performance"}
FEEDBACK_RELIABILITY = 0.8       # reliability of provider feedback vs. 1.0 for verification
PERF_PRIOR_N = 2                 # pseudo-observations pulling S_perf towards 0.5
STATUS = {-1: "rejected", 0: "none", 1: "claimed", 2: "verified", 3: "demonstrated"}


def semantic_coverage(claimed, tasks):
    """S_skill: mean over required skills of the best skill-graph match among claimed skills."""
    cs = np.stack([(claimed * SIM[s][None, :]).max(1) for s in range(N_SKILLS)], axis=1)  # [N,S]
    return req_norm(tasks) @ cs.T


def static_components(w: World, tasks: Tasks):
    """Components that do not depend on platform state."""
    cov, zone_ok = avail_matrices(w, tasks)
    ry = tasks.req_years[:, None].astype(float)
    exp = np.where(ry == 0, 1.0, np.minimum(1.0, w.years[None, :] / np.maximum(ry, 1e-9)))
    cat_ok = w.pref_cats[:, tasks.cat].T
    pay = np.minimum(1.0, tasks.pay[:, None] / w.min_pay[None, :])
    pref = np.where(cat_ok, 1.0, 0.2) * pay
    mand_claim = np.ones_like(cov, dtype=bool)
    for t in np.where(tasks.mand >= 0)[0]:
        mand_claim[t] = w.claimed[:, tasks.mand[t]]
    return dict(
        skill_raw=semantic_coverage(w.claimed, tasks),
        skill_eff=semantic_coverage(w.claimed & ~w.rejected, tasks),
        exp=exp, pref=pref, avail=cov * zone_ok,
        gate_core=(cov >= 0.999) & zone_ok, gate_mand_claim=mand_claim)


class Platform:
    """Holds the dynamic worker profile (evidence-based proficiency, status, performance)."""

    def __init__(self, world: World, lam: float = 0.3):
        self.w, self.lam = world, lam
        N = len(world.years)
        self.p = world.p_ver.copy()                      # evidence-based proficiency
        self.ev = world.verified.copy()                  # has any evidence
        self.status = np.where(world.rejected, -1,
                      np.where(world.verified, 2, np.where(world.claimed, 1, 0)))
        self.perf_sum = np.zeros((N, N_CLUSTERS))
        self.perf_n = np.zeros((N, N_CLUSTERS))

    # ---- profile update (Eq. p_new = (1-lam)p_old + lam*E) -------------------------------
    def record_outcome(self, worker, task_cat, req_idx, quality):
        if self.lam <= 0:
            return
        lam_eff = self.lam * FEEDBACK_RELIABILITY
        for s in req_idx:
            if self.ev[worker, s]:
                self.p[worker, s] = (1 - lam_eff) * self.p[worker, s] + lam_eff * quality
            else:
                self.p[worker, s] = quality                # first evidence initialises
                self.ev[worker, s] = True
            if quality >= 0.6 and self.p[worker, s] >= 0.6:
                self.status[worker, s] = 3                 # demonstrated
        self.perf_sum[worker, task_cat] += quality
        self.perf_n[worker, task_cat] += 1

    # ---- components ---------------------------------------------------------------------
    def dynamic_components(self, tasks: Tasks):
        ver = req_norm(tasks) @ np.where(self.ev, self.p, 0.0).T
        perf = ((self.perf_sum + 0.5 * PERF_PRIOR_N) / (self.perf_n + PERF_PRIOR_N))[:, tasks.cat].T
        mand_ver = np.ones_like(ver, dtype=bool)
        for t in np.where(tasks.mand >= 0)[0]:
            mand_ver[t] = self.status[:, tasks.mand[t]] >= 2
        return dict(ver=ver, perf=perf, gate_mand_ver=mand_ver)

    def components(self, tasks: Tasks):
        c = static_components(self.w, tasks)
        c.update(self.dynamic_components(tasks))
        return c

    @staticmethod
    def stack(c, use_verification=True):
        skill = c["skill_eff"] if use_verification else c["skill_raw"]
        return np.stack([skill, c["ver"], c["exp"], c["pref"], c["avail"], c["perf"]], axis=-1)

    @staticmethod
    def gate(c, use_verification=True):
        return c["gate_core"] & (c["gate_mand_ver"] if use_verification else c["gate_mand_claim"])

    def score(self, tasks, weights, use_gate=True, use_verification=True, comps=None):
        c = comps or self.components(tasks)
        m = self.stack(c, use_verification) @ np.asarray(weights)
        if use_gate:
            m = np.where(self.gate(c, use_verification), m, -np.inf)
        return m

    # ---- recommendations + explanations --------------------------------------------------
    def recommend(self, tasks, t_idx, weights, k=5):
        sub = tasks.subset(t_idx)
        m = self.score(sub, weights)[0]
        order = np.argsort(-m)[:k]
        return [(int(i), float(m[i])) for i in order if np.isfinite(m[i])]

    def explain(self, tasks, t_idx, worker, weights):
        sub = tasks.subset(t_idx)
        c = self.components(sub)
        vec = self.stack(c)[0, worker]
        weights = np.asarray(weights)
        contrib = weights * vec
        factors = [dict(name=FACTOR_LABELS[n], value=float(vec[i]), weight=float(weights[i]),
                        contribution=float(contrib[i])) for i, n in enumerate(COMP_NAMES)]
        blocking = []
        if not c["gate_core"][0, worker]:
            blocking.append("availability/location not covered")
        if not c["gate_mand_ver"][0, worker]:
            blocking.append(f"missing verified evidence for mandatory skill "
                            f"'{SKILL_NAMES[sub.mand[0]]}'")
        req = np.where(sub.req[0])[0]
        unverified = [SKILL_NAMES[s] for s in req if self.status[worker, s] == 1]
        verified = [SKILL_NAMES[s] for s in req if self.status[worker, s] in (2, 3)]
        ranked = sorted(factors, key=lambda f: -f["contribution"])
        top = [f["name"] for f in ranked[:2] if f["contribution"] > 0]
        weak = [f["name"] for f in ranked[-2:] if f["value"] < 0.5]
        text = (f"Score {contrib.sum():.3f}. Strongest factors: {', '.join(top) or 'none'}."
                + (f" Weak factors: {', '.join(weak)}." if weak else "")
                + (f" Verified required skills: {', '.join(verified)}." if verified else "")
                + (f" Claimed but unverified: {', '.join(unverified)}." if unverified else "")
                + (f" BLOCKING: {'; '.join(blocking)}." if blocking else ""))
        return dict(score=float(contrib.sum()), factors=factors, blocking=blocking,
                    verified=verified, unverified=unverified, text=text)


# ---- two-sided stable matching (task-proposing deferred acceptance) ------------------------
def deferred_acceptance(task_scores, worker_pref, max_list=50):
    """Each task proposes down its score-ordered list; each worker keeps the proposal
    with the highest worker_pref. Returns {task: worker}. -inf entries are unacceptable."""
    T, N = task_scores.shape
    lists = []
    for t in range(T):
        o = np.argsort(-task_scores[t])[:max_list]
        lists.append([int(w) for w in o if np.isfinite(task_scores[t, w])
                      and np.isfinite(worker_pref[t, w])])
    nxt, held, free = [0] * T, {}, list(range(T))
    while free:
        t = free.pop()
        while nxt[t] < len(lists[t]):
            wk = lists[t][nxt[t]]
            nxt[t] += 1
            cur = held.get(wk)
            if cur is None:
                held[wk] = t
                break
            if worker_pref[t, wk] > worker_pref[cur, wk]:
                held[wk] = t
                free.append(cur)
                break
    return {t: wk for wk, t in held.items()}

"""Synthetic worker/task world with hidden ground truth.

The platform never sees `prof` (true proficiency). It only sees claims,
verification evidence, availability/preferences, and noisy task outcomes.
Ground truth (relevance, gain) is computed from `prof`, so a system can only
score well by inferring true capability from what the platform can observe.
"""
from dataclasses import dataclass
import numpy as np
from .taxonomy import (N_SKILLS, N_CLUSTERS, CLUSTER_OF, CLUSTER_SKILLS)

N_SLOTS = 21   # 7 days x 3 shifts
N_ZONES = 5


@dataclass
class Params:
    n_workers: int = 600
    p_group_b: float = 0.35
    inflator_rate: float = 0.35      # share of workers who add false skill claims
    verif_access_a: float = 0.80     # P(worker submits evidence for a claim), group A
    verif_access_b: float = 0.50     # same, group B (simulated access gap)
    tpr: float = 0.92                # verifier accepts a true skill
    fpr: float = 0.08                # verifier accepts a false claim
    n_epochs: int = 6                # epochs 0..5 (epoch 0 = cold start)
    tasks_per_epoch: int = 100
    logs_per_epoch: int = 300        # historical hires per epoch (claim-based hiring)
    outcome_noise: float = 0.12
    mandatory_rate: float = 0.20


@dataclass
class World:
    group: np.ndarray        # [N] 0 = A, 1 = B (never used as a feature)
    prof: np.ndarray         # [N,S] TRUE proficiency (hidden)
    has: np.ndarray          # [N,S] bool true skill
    claimed: np.ndarray      # [N,S] bool self-reported skills
    inflator: np.ndarray     # [N] bool
    years: np.ndarray        # [N]
    pref_cats: np.ndarray    # [N,C] bool
    min_pay: np.ndarray      # [N]
    avail: np.ndarray        # [N,21] bool
    zone: np.ndarray         # [N]
    radius: np.ndarray       # [N]
    submit: np.ndarray       # [N,S] bool evidence submitted
    verified: np.ndarray     # [N,S] bool verifier accepted
    rejected: np.ndarray     # [N,S] bool verifier rejected
    p_ver: np.ndarray        # [N,S] verified proficiency estimate (0 if not verified)


@dataclass
class Tasks:
    cat: np.ndarray          # [T]
    req: np.ndarray          # [T,S] bool
    diff: np.ndarray         # [T] difficulty threshold
    pay: np.ndarray          # [T]
    slots: np.ndarray        # [T,21] bool
    zone: np.ndarray         # [T]
    req_years: np.ndarray    # [T]
    mand: np.ndarray         # [T] skill index needing verified evidence, -1 if none

    def __len__(self):
        return len(self.cat)

    def subset(self, idx):
        idx = np.atleast_1d(idx)
        return Tasks(*(getattr(self, f)[idx] for f in
                       ("cat", "req", "diff", "pay", "slots", "zone", "req_years", "mand")))


def generate_world(rng, p: Params) -> World:
    N, S, C = p.n_workers, N_SKILLS, N_CLUSTERS
    group = (rng.random(N) < p.p_group_b).astype(int)
    main = rng.integers(0, C, N)
    second = rng.integers(0, C, N)
    has_second = (rng.random(N) < 0.6) & (second != main)
    wc = np.zeros((N, C), bool)
    wc[np.arange(N), main] = True
    wc[np.arange(N)[has_second], second[has_second]] = True
    pool = wc[:, CLUSTER_OF]                                  # [N,S]
    has = pool & (rng.random((N, S)) < 0.70)
    prof = np.where(has, np.clip(rng.beta(5, 2.5, (N, S)), 0.2, 1.0), 0.0)
    years = np.clip(rng.gamma(2.0, 1.5, N), 0, 10)

    inflator = rng.random(N) < p.inflator_rate
    claimed = has & (rng.random((N, S)) < 0.92)
    for i in np.where(inflator)[0]:
        cand = np.where(~has[i])[0]
        wgt = np.where(pool[i, cand], 3.0, 1.0)
        k = min(int(rng.integers(2, 5)), len(cand))
        chosen = rng.choice(cand, size=k, replace=False, p=wgt / wgt.sum())
        claimed[i, chosen] = True

    pref = wc.copy() | (rng.random((N, C)) < 0.12)
    min_pay = rng.uniform(10, 28, N)
    avail = rng.random((N, N_SLOTS)) < rng.uniform(0.35, 0.8, N)[:, None]
    zone = rng.integers(0, N_ZONES, N)
    radius = rng.integers(2, 4, N)

    access = np.where(group == 1, p.verif_access_b, p.verif_access_a)
    submit = claimed & (rng.random((N, S)) < access[:, None])
    u = rng.random((N, S))
    verified = submit & np.where(has, u < p.tpr, u < p.fpr)
    rejected = submit & ~verified
    est = np.where(has, np.clip(prof + rng.normal(0, 0.08, (N, S)), 0.05, 1.0),
                   rng.uniform(0.4, 0.7, (N, S)))
    p_ver = np.where(verified, est, 0.0)
    return World(group, prof, has, claimed, inflator, years, pref, min_pay, avail,
                 zone, radius, submit, verified, rejected, p_ver)


def generate_tasks(rng, n, p: Params) -> Tasks:
    cat = rng.integers(0, N_CLUSTERS, n)
    req = np.zeros((n, N_SKILLS), bool)
    mand = -np.ones(n, int)
    for t in range(n):
        k = int(rng.integers(2, 4))
        sk = rng.choice(CLUSTER_SKILLS[cat[t]], size=k, replace=False)
        req[t, sk] = True
        if rng.random() < p.mandatory_rate:
            mand[t] = sk[0]
    slots = np.zeros((n, N_SLOTS), bool)
    for t in range(n):
        slots[t, rng.choice(N_SLOTS, size=int(rng.integers(1, 3)), replace=False)] = True
    return Tasks(cat, req, rng.uniform(0.35, 0.75, n), rng.uniform(12, 32, n), slots,
                 rng.integers(0, N_ZONES, n), rng.integers(0, 6, n), mand)


def req_norm(tasks: Tasks):
    return tasks.req / tasks.req.sum(1, keepdims=True)


def avail_matrices(w: World, tasks: Tasks):
    """coverage [T,N] in [0,1] and zone_ok [T,N] bool (stated by workers, assumed truthful)."""
    cov = (tasks.slots.astype(float) @ w.avail.T.astype(float)) / tasks.slots.sum(1, keepdims=True)
    zone_ok = np.abs(tasks.zone[:, None] - w.zone[None, :]) <= w.radius[None, :]
    return cov, zone_ok


def pref_ok_matrix(w: World, tasks: Tasks):
    cat_ok = w.pref_cats[:, tasks.cat].T                       # [T,N]
    pay_ok = tasks.pay[:, None] >= w.min_pay[None, :]
    return cat_ok & pay_ok


def ground_truth(w: World, tasks: Tasks):
    comp = req_norm(tasks) @ w.prof.T                          # true mean proficiency on required skills
    cov, zone_ok = avail_matrices(w, tasks)
    avail_ok = cov >= 0.999
    pref_ok = pref_ok_matrix(w, tasks)
    relevant = (comp >= tasks.diff[:, None]) & avail_ok & zone_ok & pref_ok
    gain = comp * avail_ok * zone_ok * np.where(pref_ok, 1.0, 0.4)
    return dict(comp=comp, relevant=relevant, gain=gain)

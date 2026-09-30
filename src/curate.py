"""Curation reachability on RewardBench v1.

A curator keeps part of the benchmark and scores every model on what is left.
Keeping a fraction f of the benchmark's official weight means item weights
w with 0 <= w_i <= w0_i / f and sum(w) = 1. For each model m we find f*_m, the
largest keep-fraction at which some curation puts m strictly first:

    max t  s.t.  sum_i w_i (R[m,i] - R[k,i]) >= t  for every other model k,
                 0 <= w_i <= w0_i / f,  sum_i w_i = 1   (+ section masses 1/4
                 in the stratified variant),

and m is crownable at f when t* > 0. Feasibility only gets easier as f falls,
so f*_m comes from bisection. Every LP answer is then rounded to a real item
subset (items at or near their upper bound), rescored with the official
weights renormalised over that subset, and kept only if m is still strictly
first. The keep-fraction we report is the rounded subset's, not the LP's.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load, load_rb2  # noqa: E402

RB2 = "rb2" in sys.argv[1:]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def solve(R, w0, sec_idx, m, f, stratified):
    """LP for model m at keep-fraction f. Returns (t*, w) or (None, None)."""
    M, N = R.shape
    others = [k for k in range(M) if k != m]
    D = R[others] - R[m]                       # (M-1, N): want D @ w <= -t
    # variables: [w (N), t]
    A_ub = np.hstack([D, np.ones((len(others), 1))])
    b_ub = np.zeros(len(others))
    A_eq = [np.append(np.ones(N), 0.0)]
    b_eq = [1.0]
    if stratified:
        for s in sec_idx:
            row = np.zeros(N + 1)
            row[s] = 1.0
            A_eq.append(row)
            b_eq.append(w0[s].sum())
    bounds = [(0.0, w0[i] / f) for i in range(N)] + [(None, None)]
    c = np.zeros(N + 1)
    c[-1] = -1.0
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                bounds=bounds, method="highs")
    if r.status != 0:
        return None, None
    return -r.fun, r.x[:N]


def round_subset(w, w0, f, sec_idx, stratified, thresh):
    """Turn LP weights into a kept item set and its official-weight scores.
    Items with LP weight above thresh * (their upper bound) are kept."""
    keep = w > thresh * w0 / f
    if stratified:
        wt = np.zeros_like(w0)
        for s in sec_idx:
            k = s[keep[s]]
            if len(k) == 0:
                return keep, None
            wt[k] = w0[k] / w0[k].sum() * w0[s].sum()
    else:
        wt = np.where(keep, w0, 0.0)
        wt = wt / wt.sum()
    return keep, wt


GRID = [1.0, 0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4,
        0.3, 0.2, 0.1, 0.05, 0.02, 0.01]


def crown(R, w0, sec_idx, m, stratified, iters=10):
    """Largest verified keep-fraction at which model m is strictly first.
    Scan a coarse grid from f = 1 downwards, then bisect between the first
    grid point that verifies and the one above it."""
    def verified(f):
        t, w = solve(R, w0, sec_idx, m, f, stratified)
        if t is None or t <= 1e-9:
            return None
        best = None
        for thresh in (0.5, 1e-9):            # near-integral first, then all
            keep, wt = round_subset(w, w0, f, sec_idx, stratified, thresh)
            if wt is None:
                continue
            s = R @ wt
            rest = np.delete(s, m).max()
            if s[m] > rest + 1e-12:
                v = dict(f=float(w0[keep].sum()), n_items=int(keep.sum()),
                         margin=float(s[m] - rest),
                         keep=np.flatnonzero(keep).tolist())
                if best is None or v["f"] > best["f"]:
                    best = v
        return best

    prev, best = None, None
    for g in GRID:
        best = verified(g)
        if best is not None:
            break
        prev = g
    if best is None:
        return None
    if prev is None:
        return best
    a, b = g, prev
    for _ in range(iters):
        mid = (a + b) / 2
        v = verified(mid)
        if v is not None:
            a = mid
            if v["f"] > best["f"]:
                best = v
        else:
            b = mid
    return best


def main(pool_top=None):
    d = load_rb2() if RB2 else load()
    R, w0, names = d["R"], d["w0"], d["names"]
    official = R @ w0
    order = np.argsort(-official)
    if pool_top:
        order = order[:pool_top]
        R = R[order]
        names = [names[j] for j in order]
        official = official[order]
        order = np.arange(len(order))
    sec_idx = [np.flatnonzero(d["section"] == s)
               for s in sorted(set(d["section"]))]
    rows = []
    for rank, m in enumerate(order, 1):
        row = dict(model=names[m], official=float(official[m]),
                   official_rank=rank)
        for strat in (False, True):
            r = crown(R, w0, sec_idx, m, strat)
            key = "strat" if strat else "free"
            row[key] = None if r is None else {k: v for k, v in r.items()
                                              if k != "keep"}
            if r is not None and rank <= 25:
                row[key + "_keep"] = r["keep"]
        rows.append(row)
        print("%3d %-55s %.3f  free %s  strat %s" % (
            rank, names[m][:55], official[m],
            "-" if row["free"] is None else "%.3f" % row["free"]["f"],
            "-" if row["strat"] is None else "%.3f" % row["strat"]["f"]),
            flush=True)
    tag = ("rb2_" if RB2 else "") + ("top%d" % pool_top if pool_top else "all")
    json.dump(rows, open(os.path.join(OUT, "crown_%s.json" % tag), "w"))


if __name__ == "__main__":
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    main(int(nums[0]) if nums else None)

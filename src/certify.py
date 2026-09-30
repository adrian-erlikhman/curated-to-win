"""Certified evaluator selection.

1. Share certificate. By Prop. 1 the official winner stays first under any
   curation that removes a share below its gap to the runner-up; the LP
   results give the smallest verified removal that dethrones it (upper end).
2. Item certificate. For challenger c to overtake leader a, the removed items
   must satisfy sum_{i removed} w_i (r_ai - r_ci) > g_ac. The fewest items that
   can do this is found exactly by taking the largest w_i (r_ai - r_ci) first;
   the leader is certified against removing fewer than min_c of that count.
3. Plausible-winner sets. At curation budget delta, a model can be crowned
   only if its gap <= delta (Prop. 1) and can certainly be crowned if its
   verified LP cost <= delta, which brackets the set of possible winners.
4. Certificates on small validation sets. Draw n-item validation sets as in
   benign.py, pick the winner on each, and compute the certificate on that
   set (its top-two gap). Report how often the pick equals the full-benchmark
   winner, split by whether the set's certificate exceeds a budget.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load, load_rb2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def best_rows(tag):
    rows = json.load(open(os.path.join(RES, "crown_%sall.json" % tag)))
    for r in rows:
        if r["strat"] is not None and (r["free"] is None or r["strat"]["f"] > r["free"]["f"]):
            r["free"] = dict(r["strat"])
    return rows


def item_certificate(R, w0, S, lead):
    counts = {}
    for c in range(len(S)):
        if c == lead:
            continue
        g = S[lead] - S[c]
        gain = np.sort(w0 * (R[lead] - R[c]))[::-1]
        cum = np.cumsum(gain)
        k = int(np.searchsorted(cum, g, side="right")) + 1   # need cum > g
        counts[c] = k if k <= len(gain) and cum[min(k, len(cum)) - 1] > g else None
    return counts


def run(tag, d, draws=2000, seed=20260930):
    R, w0, names = d["R"], d["w0"], d["names"]
    S = R @ w0
    order = np.argsort(-S)
    lead, second = int(order[0]), int(order[1])
    rows = best_rows(tag)
    cost = {r["model"]: (1 - r["free"]["f"]) if r["free"] else None for r in rows}
    items = {r["model"]: (R.shape[1] - r["free"]["n_items"]) if r["free"] else None for r in rows}
    ic = item_certificate(R, w0, S, lead)
    k_min = min(v for v in ic.values() if v is not None)
    c_min = min(ic, key=lambda c: ic[c] if ic[c] is not None else 1e9)
    lp_min = min(cost[names[c]] for c in range(len(S)) if c != lead and cost[names[c]] is not None)
    lp_items = min(items[names[c]] for c in range(len(S)) if c != lead and items[names[c]] is not None)

    budgets = [0.005, 0.01, 0.02, 0.05, 0.10]
    plaus = []
    for dlt in budgets:
        upper = int(sum(S[lead] - S[c] <= dlt for c in range(len(S))))        # includes leader
        lower = 1 + int(sum(cost[names[c]] is not None and cost[names[c]] <= dlt
                            for c in range(len(S)) if c != lead))
        plaus.append(dict(delta=dlt, lower=lower, upper=upper))

    # certificates on small validation sets
    rng = np.random.default_rng(seed)
    p = w0 / w0.sum()
    val = []
    for n in (100, 200, 500):
        picks, radii = [], []
        for _ in range(draws):
            idx = rng.choice(len(p), size=n, replace=True, p=p)
            s = R[:, idx].mean(axis=1)
            top = np.flatnonzero(s == s.max())
            pick = int(rng.choice(top))                  # random tie-break, as in benign.py
            picks.append(pick == lead)
            rest = np.delete(s, pick)
            radii.append(float(s[pick] - rest.max()))   # certificate on this set (0 on a tie)
        picks, radii = np.array(picks), np.array(radii)
        rowv = dict(n=n, p_correct=float(picks.mean()))
        for thr in (0.01, 0.02, 0.05):
            m = radii > thr
            rowv["cert_%d" % round(thr * 100)] = dict(
                share_certified=float(m.mean()),
                p_correct_certified=float(picks[m].mean()) if m.any() else None,
                p_correct_uncertified=float(picks[~m].mean()) if (~m).any() else None)
        val.append(rowv)
    return dict(leader=names[lead], runner_up=names[second],
                gap=float(S[lead] - S[second]), lp_min_cost=float(lp_min), lp_min_items=int(lp_items),
                item_cert=int(k_min), item_cert_model=names[c_min],
                plausible=plaus, validation=val)


if __name__ == "__main__":
    out = {"rb1": run("", load()), "rb2": run("rb2_", load_rb2())}
    json.dump(out, open(os.path.join(RES, "certify.json"), "w"), indent=1)
    for k, v in out.items():
        print(k, "gap %.4f  LP min %.4f (%d items)  item certificate: fewer than %d items (%s)" % (
            v["gap"], v["lp_min_cost"], v["lp_min_items"], v["item_cert"], v["item_cert_model"]))
        for pl in v["plausible"]:
            print("   delta %.3f: plausible winners %d..%d" % (pl["delta"], pl["lower"], pl["upper"]))
        for r in v["validation"]:
            print("   n=%d p(correct)=%.2f" % (r["n"], r["p_correct"]),
                  {k2: {kk: (round(vv, 2) if vv is not None else None) for kk, vv in r[k2].items()}
                   for k2 in r if k2.startswith("cert")})

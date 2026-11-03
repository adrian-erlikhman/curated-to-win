"""Item certificates under the official scoring code, as a check on the
paper's renormalized-weight operator.

The paper rescores a curation by renormalizing the original item weights
(S^K = sum_K w0 r / f). RewardBench's own code instead recomputes each
subset's accuracy over the items that remain and keeps the fixed subset
weights (EXAMPLE_COUNTS within a section, the four sections averaged);
RewardBench 2's code averages its subset accuracies. Under that operator an
item's effect depends on how many of its subset remain, so Prop. 2 does not
apply directly. The exact minimum is still easy to find: within a subset,
for a fixed number k of removals, the best k items for a challenger are the
k with the largest r_a - r_c (the denominator n_s - k is fixed), and a
small dynamic program over subsets picks how many to take from each.

    python src/check_operator.py
"""
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import EXAMPLE_COUNTS, SECTIONS, SECTION_OF, load, load_rb2  # noqa: E402

KMAX = 60


def subset_coef(d, rb2):
    """Each subset's coefficient in the official score: score = sum_s c_s acc_s."""
    subs = sorted(set(d["subset"]))
    if rb2:
        return {s: 1 / len(subs) for s in subs}
    c = {}
    for sec, members in SECTIONS.items():
        total = sum(EXAMPLE_COUNTS[s] for s in members)
        for s in members:
            c[s] = 0.25 * EXAMPLE_COUNTS[s] / total
    return c


def official(R, d, rb2, keep=None):
    c = subset_coef(d, rb2)
    keep = np.ones(R.shape[1], bool) if keep is None else keep
    S = np.zeros(R.shape[0])
    for s, cs in c.items():
        m = (d["subset"] == s) & keep
        S += cs * R[:, m].mean(axis=1)
    return S


def min_removals(R, d, rb2, a, ch, kmax=KMAX):
    """Fewest removals after which challenger ch strictly beats a, under the
    official operator; returns (k, removal per subset) or (None, None)."""
    c = subset_coef(d, rb2)
    subs = sorted(c)
    best = {}            # subset -> array over k of the challenger-minus-leader term
    for s in subs:
        m = d["subset"] == s
        diff = R[a, m] - R[ch, m]
        order = np.sort(diff)[::-1]
        n, D = len(diff), diff.sum()
        ks = np.arange(0, min(kmax, n - 1) + 1)
        removed = np.concatenate([[0], np.cumsum(order)])[ks]
        best[s] = c[s] * (D - removed) / (n - ks)            # leader's edge left in s
    # DP over subsets: lowest total leader edge for each total k
    INF = np.inf
    f = np.full(kmax + 1, INF)
    f[0] = 0.0
    choice = []
    for s in subs:
        g = np.full(kmax + 1, INF)
        arg = np.zeros(kmax + 1, int)
        for k in range(kmax + 1):
            if f[k] == INF:
                continue
            for j in range(len(best[s])):
                if k + j > kmax:
                    break
                v = f[k] + best[s][j]
                if v < g[k + j]:
                    g[k + j], arg[k + j] = v, j
        choice.append(arg)
        f = g
    hits = np.nonzero(f < -1e-12)[0]
    if not len(hits):
        return None, None
    k = int(hits[0])
    per = {}
    for s, arg in zip(reversed(subs), reversed(choice)):
        j = int(arg[k])
        if j:
            per[s] = j
        k -= j
    return int(hits[0]), per


def removal_mask(R, d, a, ch, per):
    keep = np.ones(R.shape[1], bool)
    for s, j in per.items():
        idx = np.nonzero(d["subset"] == s)[0]
        diff = R[a, idx] - R[ch, idx]
        keep[idx[np.argsort(-diff, kind="stable")[:j]]] = False
    return keep


def item_id(d, i, rb2):
    """How to find item i again: RewardBench 2 by (subset, id); RewardBench
    by its position in the per-item result files, with a hash of the chosen
    response's text."""
    if rb2:
        return {"subset": str(d["subset"][i]), "id": str(d["ids"][i])}
    return {"subset": str(d["subset"][i]), "position": int(i), "hash_chosen": d["hash"][i]}


def report(tag, d, rb2):
    R, w0, names = d["R"], d["w0"], d["names"]
    S_paper = R @ w0
    S_off = official(R, d, rb2)
    a = int(np.argmax(S_off))
    order = np.argsort(-S_off)
    ch = int(order[1])
    print("\n== %s: %d models, %d items" % (tag, *R.shape))
    print("max |official - R@w0| over all models: %.2e" % np.abs(S_off - S_paper).max())
    print("leader %s  %.2f; runner-up %s  %.2f" % (names[a], 100 * S_off[a], names[ch], 100 * S_off[ch]))
    # the paper's item certificate (Prop. 2, renormalized weights), then rescored the official way
    gain = w0 * (R[a] - R[ch])
    idx = np.argsort(-gain, kind="stable")
    cum = np.cumsum(gain[idx])
    k_paper = int(np.searchsorted(cum, S_paper[a] - S_paper[ch], side="right")) + 1
    keep_p = np.ones(R.shape[1], bool)
    keep_p[idx[:k_paper]] = False
    S_k = official(R, d, rb2, keep_p)
    top = np.argsort(-S_k)[:3]
    print("paper's %d items removed, official rescoring: %s" %
          (k_paper, ", ".join("%s %.3f" % (names[j], 100 * S_k[j]) for j in top)))
    paper_set = dict(k=k_paper, items=[item_id(d, int(i), rb2) for i in idx[:k_paper]],
                     first_official=names[int(top[0])],
                     official_scores={names[int(j)]: round(100 * S_k[j], 4) for j in top})
    # exact item certificate under the official operator, over all challengers
    res = []
    for c in order[1:]:
        k, per = min_removals(R, d, rb2, a, int(c))
        if k is not None:
            res.append((k, int(c), per))
    res.sort(key=lambda t: t[0])
    k_star = res[0][0]
    print("official-operator item certificate: leader safe vs any %d, %d flip it" % (k_star - 1, k_star))
    best = []
    for k, c, per in res[:5]:
        keep = removal_mask(R, d, a, c, per)
        S_k = official(R, d, rb2, keep)
        first = int(np.argmax(S_k))
        print("  %-48s k=%d %s  -> first after removal: %s" % (names[c], k, per, names[first]))
        best.append(dict(challenger=names[c], k=k, per_subset={str(s): j for s, j in per.items()},
                         first_after=names[first],
                         items=[item_id(d, int(i), rb2) for i in np.nonzero(~keep)[0]],
                         official_scores={names[a]: round(100 * S_k[a], 4),
                                          names[c]: round(100 * S_k[c], 4)}))
    return dict(leader=names[a], runner_up=names[ch], k_paper=k_paper, paper_set=paper_set,
                k_official=k_star, official_best=best)


def rb2_official_check(d, n=8):
    """Our five-subset scores against RewardBench 2's published subset scores
    (which add Ties) for our top n models. A model ranked below n by us cannot
    reach the leader with Ties: its published score is at most
    (5 * ours + 100) / 6."""
    URL = "https://huggingface.co/datasets/allenai/reward-bench-2-results/resolve/main/eval-set/"
    R, w0, names = d["R"], d["w0"], d["names"]
    S = R @ w0
    order = np.argsort(-S)
    rows = []
    for j in order[:n]:
        off = json.load(urllib.request.urlopen(URL + names[j] + ".json", timeout=60))
        subs = ["Factuality", "Precise IF", "Math", "Safety", "Focus", "Ties"]
        rows.append(dict(model=names[j], ours=100 * S[j],
                         published={s: 100 * off[s] for s in subs},
                         with_ties=100 * np.mean([off[s] for s in subs])))
    best = sorted(rows, key=lambda r: -r["with_ties"])
    bound = (5 * 100 * S[order[n]] + 100) / 6
    for r in rows:
        print("%-48s ours %.2f  with Ties %.2f" % (r["model"], r["ours"], r["with_ties"]))
    print("with Ties: first %s, lead %.2f over %s; models below our top %d reach at most %.2f"
          % (best[0]["model"], best[0]["with_ties"] - best[1]["with_ties"], best[1]["model"], n, bound))
    return dict(rows=rows, first_with_ties=best[0]["model"], runner_up_with_ties=best[1]["model"],
                lead_with_ties=best[0]["with_ties"] - best[1]["with_ties"],
                bound_below_top_n=bound, same_winner=best[0]["model"] == names[order[0]]
                and best[0]["with_ties"] > bound)


if __name__ == "__main__":
    d2 = load_rb2()
    out = {"rb1": report("RewardBench", load(), False),
           "rb2": report("RewardBench 2", d2, True)}
    print("\n== RewardBench 2 published scores (with Ties) for our top models")
    out["rb2_ties"] = rb2_official_check(d2)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "results", "check_operator.json"), "w"), indent=1,
              default=lambda o: o.item())       # numpy scalars

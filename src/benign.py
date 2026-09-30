"""Benign curation on RewardBench v1: no adversary, just small or narrow
validation sets.

1. Random validation sets. Draw n items with probability proportional to
   their official weight (with replacement), so a set's plain mean is an
   unbiased estimate of the official score. Report how often the official #1
   is still first, how many distinct models ever come first, and the Kendall
   tau of the official top 10 on the drawn set.
2. Natural curations. Score on each section alone, each subset alone, and
   with each section left out; report who wins and where that winner sits
   on the official leaderboard.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load, load_rb2  # noqa: E402

RB2 = "rb2" in sys.argv[1:]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def main(draws=2000, seed=20260930):
    d = load_rb2() if RB2 else load()
    R, w0, names = d["R"], d["w0"], d["names"]
    official = R @ w0
    rank = np.empty(len(names), int)
    rank[np.argsort(-official)] = np.arange(1, len(names) + 1)
    top = int(np.argmax(official))
    top10 = np.argsort(-official)[:10]
    rng = np.random.default_rng(seed)
    p = w0 / w0.sum()

    random_sets = []
    for n in (25, 50, 100, 200, 500, 1000):
        wins, taus, winners = 0, [], set()
        for _ in range(draws):
            idx = rng.choice(len(p), size=n, replace=True, p=p)
            s = R[:, idx].mean(axis=1)
            best = np.flatnonzero(s == s.max())
            w = rng.choice(best)               # break exact ties at random
            wins += int(w == top)
            winners.add(int(w))
            taus.append(kendalltau(official[top10], s[top10]).statistic)
        random_sets.append(dict(
            n=n, p_top_kept=wins / draws, distinct_winners=len(winners),
            worst_winner_rank=int(max(rank[list(winners)])),
            tau_top10_median=float(np.nanmedian(taus)),
            tau_top10_p10=float(np.nanpercentile(taus, 10))))
        print(random_sets[-1], flush=True)

    def winner(weights):
        s = R @ (weights / weights.sum())
        tied = np.flatnonzero(s >= s.max() - 1e-12)
        j = int(tied[np.argmin(rank[tied])])   # best official rank among ties
        return dict(winner=names[j], official_rank=int(rank[j]),
                    score=float(s[j]), n_tied=int(len(tied)))

    natural = {}
    for sec in sorted(set(d["section"])):
        m = d["section"] == sec
        natural["only " + sec] = winner(np.where(m, w0, 0))
        natural["without " + sec] = winner(np.where(~m, w0, 0))
    for sub in sorted(set(d["subset"])):
        natural["subset " + sub] = winner(np.where(d["subset"] == sub, w0, 0))
    for k, v in natural.items():
        print("%-32s #%-3d tied %-3d %s" % (k, v["official_rank"],
                                            v["n_tied"], v["winner"]))

    json.dump(dict(official_top=names[top], n_models=len(names),
                   random_sets=random_sets, natural=natural),
              open(os.path.join(OUT, "benign_rb2.json" if RB2 else "benign.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

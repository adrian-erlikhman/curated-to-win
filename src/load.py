"""Load the compacted RewardBench v1 per-item results into one matrix.

R[m, i] is model m's score on item i (1 = preferred the chosen response,
0 = preferred the rejected one; a few generative judges record 0.5 for ties).
w0[i] is item i's weight in the official RewardBench score, so that
R @ w0 reproduces each model's leaderboard number: the mean of four section
scores, each a count-weighted mean of subset accuracies, with math-prm
upweighted to 984 examples as in rewardbench/constants.py.
"""
import collections
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100, "alpacaeval-length": 95, "alpacaeval-hard": 95,
    "mt-bench-easy": 28, "mt-bench-med": 40, "mt-bench-hard": 37,
    "math-prm": 984, "refusals-dangerous": 100, "refusals-offensive": 100,
    "llmbar-natural": 100, "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92, "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46, "xstest-should-refuse": 154,
    "xstest-should-respond": 250, "donotanswer": 136,
    "hep-cpp": 164, "hep-go": 164, "hep-java": 164, "hep-js": 164,
    "hep-python": 164, "hep-rust": 164,
}
SECTIONS = {
    "Chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard",
             "mt-bench-easy", "mt-bench-med"],
    "Chat Hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
                  "llmbar-adver-GPTInst", "llmbar-adver-GPTOut",
                  "llmbar-adver-manual"],
    "Safety": ["refusals-dangerous", "refusals-offensive",
               "xstest-should-refuse", "xstest-should-respond", "donotanswer"],
    "Reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js",
                  "hep-python", "hep-rust"],
}
SECTION_OF = {s: sec for sec, subs in SECTIONS.items() for s in subs}


def official_weights(subset):
    subset = np.asarray(subset)
    n = collections.Counter(subset)
    w = np.zeros(len(subset))
    for sec, subs in SECTIONS.items():
        total = sum(EXAMPLE_COUNTS[s] for s in subs)
        for s in subs:
            w[subset == s] = 0.25 * EXAMPLE_COUNTS[s] / n[s] / total
    return w


def load():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "compact", "*.json")))
    recs = [json.load(open(f)) for f in files]
    ref = max(recs, key=lambda r: len(r["subset"]))["subset"]
    keep, dropped = [], []
    for f, r in zip(files, recs):
        (keep if r["subset"] == ref else dropped).append((f, r))
    # items must line up across files: same subset sequence, and the chosen
    # text should hash the same at each position for most models
    H = np.array([r["hash_chosen"] for _, r in keep])
    modal = [collections.Counter(H[:, i]).most_common(1)[0][0]
             for i in range(H.shape[1])]
    agree = (H == np.array(modal)).mean(axis=1)
    R = np.array([r["results"] for _, r in keep], dtype=float)
    names = [r["model"] for _, r in keep]
    types = [r.get("model_type") for _, r in keep]
    subset = np.array(ref)
    return dict(R=R, names=names, types=types, subset=subset,
                section=np.array([SECTION_OF[s] for s in subset]),
                w0=official_weights(subset), hash_agree=agree,
                dropped=[os.path.basename(f) for f, _ in dropped])


if __name__ == "__main__":
    d = load()
    R, w0 = d["R"], d["w0"]
    print("models", R.shape[0], "items", R.shape[1], "dropped", d["dropped"])
    print("hash agreement: min %.3f, models <0.9: %d" %
          (d["hash_agree"].min(), (d["hash_agree"] < 0.9).sum()))
    print("result values", np.unique(R))
    s = R @ w0
    for j in np.argsort(-s)[:10]:
        print("%.4f  %s" % (s[j], d["names"][j]))


RB2_SUBSETS = ["Factuality", "Precise IF", "Math", "Safety", "Focus"]


def load_rb2():
    """RewardBench 2, five accuracy subsets (Ties is scored differently and
    is left out). An item counts as correct when the model scores the chosen
    completion above all three rejected ones. Items are matched across models
    by (subset, id); the score is the unweighted mean of the five subset
    accuracies, so w0 is 1/5 spread evenly over each subset's items."""
    files = sorted(glob.glob(os.path.join(ROOT, "data", "compact_rb2", "*.json")))
    recs = []
    for f in files:
        r = json.load(open(f))
        if not r.get("id") or not r.get("results"):
            continue
        recs.append({(s, str(i)): float(x) for s, i, x in
                     zip(r["subset"], r["id"], r["results"])
                     if s in RB2_SUBSETS} | {"_model": r["model"],
                                              "_type": r.get("model_type")})
    keys = sorted(set.intersection(*[set(k for k in r if k[0] != "_")
                                     for r in recs if len(r) > 1000]))
    recs = [r for r in recs if all(k in r for k in keys)]
    R = np.array([[r[k] for k in keys] for r in recs])
    subset = np.array([k[0] for k in keys])
    w0 = np.zeros(len(keys))
    for s in RB2_SUBSETS:
        w0[subset == s] = 0.2 / (subset == s).sum()
    return dict(R=R, names=[r["_model"] for r in recs],
                types=[r["_type"] for r in recs], subset=subset,
                section=subset, w0=w0, dropped=[])

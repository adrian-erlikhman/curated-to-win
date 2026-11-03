"""Check rebuilt subset accuracies against RewardBench's own per-model
subset scores (allenai/reward-bench-results, eval-set/<model>.json) for
every model; check_official.py did four. Writes results/check_official_all.json."""
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://huggingface.co/datasets/allenai/reward-bench-results/resolve/main/eval-set/"
d = load()
rows, missing = [], []
for j, m in enumerate(d["names"]):
    try:
        off = json.load(urllib.request.urlopen(URL + m + ".json", timeout=60))
    except Exception as e:  # noqa: BLE001
        missing.append(dict(model=m, error=str(e)[:60]))
        continue
    diffs = [abs(d["R"][j][d["subset"] == s].mean() - off[s]) for s in set(d["subset"]) if s in off]
    rows.append(dict(model=m, subsets=len(diffs), max_abs_diff=float(max(diffs))))
out = dict(n_models=len(d["names"]), n_checked=len(rows), missing=missing,
           max_abs_diff=max(r["max_abs_diff"] for r in rows),
           n_exact=sum(r["max_abs_diff"] == 0 for r in rows),
           model_types=dict(collections.Counter(d["types"])), rows=rows)
json.dump(out, open(os.path.join(ROOT, "results", "check_official_all.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})

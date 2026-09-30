"""Download RewardBench 2 per-item results (allenai/reward-bench-2-results,
eval-set-scores/) keeping id, subset and per-item correctness only."""
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "compact_rb2")
URL = "https://huggingface.co/datasets/allenai/reward-bench-2-results/resolve/main/"
os.makedirs(OUT, exist_ok=True)

def get(path):
    dst = os.path.join(OUT, path.replace("/", "__"))
    if os.path.exists(dst):
        return path, "cached"
    err = None
    for _ in range(4):
        try:
            d = json.load(urllib.request.urlopen(URL + path, timeout=300))
            break
        except Exception as e:
            err = e
    else:
        return path, "FAIL %r" % err
    if not isinstance(d, dict) or "results" not in d:
        return path, "SKIP format"
    json.dump({k: d.get(k) for k in ("model", "model_type", "id", "subset", "results")},
              open(dst, "w"))
    return path, "ok"

rows = json.load(open(os.path.join(ROOT, "data", "filelist_rb2.json")))
with ThreadPoolExecutor(8) as ex:
    for p, s in ex.map(get, [p for p, _ in rows]):
        print(s, p, flush=True)

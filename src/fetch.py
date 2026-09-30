"""Download RewardBench v1 per-item results (allenai/reward-bench-results,
eval-set-scores/) and keep only what the analysis needs: per-item correctness,
subset, and a hash of each item's chosen/rejected text for alignment checks."""
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "compact")
URL = "https://huggingface.co/datasets/allenai/reward-bench-results/resolve/main/"
os.makedirs(OUT, exist_ok=True)

def h(s):
    return hashlib.sha1(s[-400:].encode("utf-8", "ignore")).hexdigest()[:12]

def get(path):
    dst = os.path.join(OUT, path.replace("/", "__"))
    if os.path.exists(dst):
        return path, "cached"
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(URL + path, timeout=120))
            break
        except Exception as e:
            err = e
    else:
        return path, "FAIL %r" % err
    keep = {
        "model": d.get("model"), "model_type": d.get("model_type"),
        "subset": d["subset"], "results": d["results"],
        "hash_chosen": [h(t if isinstance(t, str) else json.dumps(t)) for t in d.get("text_chosen", [])],
    }
    json.dump(keep, open(dst, "w"))
    return path, "ok"

rows = json.load(open(os.path.join(ROOT, "data", "filelist.json")))
paths = [p for p, _ in rows if "ref_free" not in p]
with ThreadPoolExecutor(8) as ex:
    for p, s in ex.map(get, paths):
        print(s, p, flush=True)

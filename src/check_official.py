"""Check rebuilt subset accuracies against RewardBench's own per-model
subset scores (allenai/reward-bench-results, eval-set/<model>.json)."""
import json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load

MODELS = ["infly/INF-ORM-Llama3.1-70B", "Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
          "openai/gpt-4o-2024-08-06", "Qwen/Qwen1.5-0.5B-Chat"]
URL = "https://huggingface.co/datasets/allenai/reward-bench-results/resolve/main/eval-set/"
d = load()
for m in MODELS:
    j = d["names"].index(m)
    off = json.load(urllib.request.urlopen(URL + m + ".json"))
    diffs = [abs(d["R"][j][d["subset"] == s].mean() - off[s])
             for s in set(d["subset"]) if s in off]
    print("%-45s %d subsets, max |diff| %.1e" % (m, len(diffs), max(diffs)))

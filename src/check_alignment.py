"""Check that RewardBench v1 per-item results line up by position across
models with different chat templates: for each item, the last words of one
model's chosen response should appear at the same position in the other
model's file, and never at a different position of the same subset.
Needs two raw files in data/ (sample.json: Qwen1.5-0.5B-Chat,
sample2.json: Skywork-Reward-Gemma-2-27B-v0.2)."""
import json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a = json.load(open(os.path.join(ROOT, "data", "sample.json")))
b = json.load(open(os.path.join(ROOT, "data", "sample2.json")))
txt = lambda t: t if isinstance(t, str) else json.dumps(t)
A = [txt(x) for x in a["text_chosen"]]; B = [txt(x) for x in b["text_chosen"]]
words = lambda s: re.findall(r"[A-Za-z]{4,}", s)
def resp(s):
    return words(re.split(r"<\|im_start\|>assistant|<start_of_turn>model|\[/INST\]|<\|assistant\|>", s)[-1])[-12:]
same = other = none = 0
for i in range(len(A)):
    wa = " ".join(resp(A[i]))
    if not wa or " ".join(resp(B[i])) == wa or wa in " ".join(words(B[i])):
        same += 1
    elif any(a["subset"][j] == a["subset"][i] and wa in " ".join(words(B[j])) for j in range(len(B))):
        other += 1
    else:
        none += 1
print("same position %d, other position %d, not found %d" % (same, other, none))

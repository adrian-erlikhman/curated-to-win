"""results/check_operator.json and results/check_official_all.json -> macros
appended to paper/numbers.tex. Run after certify_report.py."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PAPER = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper")
op = json.load(open(os.path.join(RES, "check_operator.json")))
ch = json.load(open(os.path.join(RES, "check_official_all.json")))
L = []


def mac(k, v):
    L.append("\\newcommand{\\%s}{%s}" % (k, v))


for key, P in (("rb1", "One"), ("rb2", "Two")):
    v = op[key]
    assert v["paper_set"]["first_official"] == v["runner_up"]      # Prop. 2's set also works officially
    assert v["official_best"][0]["first_after"] == v["official_best"][0]["challenger"]
    mac(P + "OffItems", v["k_official"]); mac(P + "OffItemsSafe", v["k_official"] - 1)
t = op["rb2_ties"]
assert t["same_winner"] and t["first_with_ties"] == op["rb2"]["leader"]
mac("TwoTiesLead", "%.2f" % t["lead_with_ties"])
mac("OneChecked", ch["n_exact"]); mac("OneUnpublished", len(ch["missing"]))
assert ch["n_exact"] == ch["n_checked"]                             # every published file matches exactly
mac("OneJudges", ch["model_types"].get("Generative RM", 0))
with open(os.path.join(PAPER, "numbers.tex"), "a") as fh:
    fh.write("\n".join(L) + "\n")
print("\n".join(L))

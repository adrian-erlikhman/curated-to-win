"""results/certify.json -> macros appended to paper/numbers.tex, plus
paper/table_plausible.tex and paper/table_valcert.tex. Run after report.py."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PAPER = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper")
c = json.load(open(os.path.join(RES, "certify.json")))
L = []


def mac(k, v):
    L.append("\\newcommand{\\%s}{%s}" % (k, v))


pc = lambda x: "--" if x is None else "%.0f" % (100 * x)
for key, P in (("rb1", "One"), ("rb2", "Two")):
    v = c[key]
    mac(P + "CertItems", v["item_cert"]); mac(P + "CertItemsSafe", v["item_cert"] - 1)
    mac(P + "CertGap", "%.2f" % (100 * v["gap"])); mac(P + "CertLP", "%.2f" % (100 * v["lp_min_cost"]))
    mac(P + "CertLPItems", v["lp_min_items"])
    for pl in v["plausible"]:
        k = {0.005: "Half", 0.01: "One", 0.02: "Two", 0.05: "Five", 0.1: "Ten"}[round(pl["delta"], 3)]
        mac(P + "Plaus" + k + "Lo", pl["lower"]); mac(P + "Plaus" + k + "Hi", pl["upper"])
    for r in v["validation"]:
        n = {100: "Hundred", 200: "TwoHundred", 500: "FiveHundred"}[r["n"]]
        c2 = r["cert_2"]
        mac(P + "Val" + n, pc(r["p_correct"]))
        mac(P + "Val" + n + "CertShare", pc(c2["share_certified"]))
        mac(P + "Val" + n + "Cert", pc(c2["p_correct_certified"]))
        mac(P + "Val" + n + "Uncert", pc(c2["p_correct_uncertified"]))
with open(os.path.join(PAPER, "numbers.tex"), "a") as fh:
    fh.write("\n".join(L) + "\n")

T = ["\\begin{tabular}{lccccc}", "\\toprule",
     "Budget $\\delta$ & 0.5\\% & 1\\% & 2\\% & 5\\% & 10\\%\\\\", "\\midrule"]
for key, nm in (("rb1", "RewardBench"), ("rb2", "RewardBench~2")):
    cells = ["%d" % pl["lower"] if pl["lower"] == pl["upper"] else "%d--%d" % (pl["lower"], pl["upper"])
             for pl in c[key]["plausible"]]
    T.append("%s & %s\\\\" % (nm, " & ".join(cells)))
T += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(PAPER, "table_plausible.tex"), "w").write("\n".join(T) + "\n")

T = ["\\begin{tabular}{rcccccc}", "\\toprule",
     "& \\multicolumn{3}{c}{RewardBench} & \\multicolumn{3}{c}{RewardBench~2}\\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
     "$n$ & all & cert. & not & all & cert. & not\\\\", "\\midrule"]
for i, n in enumerate((100, 200, 500)):
    cells = []
    for key in ("rb1", "rb2"):
        r = c[key]["validation"][i]; c2 = r["cert_2"]
        cells += [pc(r["p_correct"]),
                  "%s {\\scriptsize(%s)}" % (pc(c2["p_correct_certified"]), pc(c2["share_certified"])),
                  pc(c2["p_correct_uncertified"])]
    T.append("%d & %s\\\\" % (n, " & ".join(cells)))
T += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(PAPER, "table_valcert.tex"), "w").write("\n".join(T) + "\n")
print("\n".join(L))

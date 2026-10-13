"""Turn results/*.json into paper/numbers.tex, paper/table_benign.tex and
paper/fig_cost.pdf. Every number the paper quotes comes from here."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")
N_ITEMS = {"": 2985, "rb2_": 1753}


def load_rows(tag):
    """Crowning results, with the unconstrained cost taken as the better of the
    two searches: a verified section-preserving subset is also a valid
    unconstrained one, and both searches are heuristic (bisection + rounding)."""
    rows = json.load(open(os.path.join(RES, "crown_%sall.json" % tag)))
    for r in rows:
        if r["strat"] is not None and (r["free"] is None or r["strat"]["f"] > r["free"]["f"]):
            r["free"] = dict(r["strat"])
    return rows


def summarize(tag):
    rows = load_rows(tag)
    top = rows[0]["official"]
    out = dict(n_models=len(rows))
    for key in ("free", "strat"):
        gaps, costs = [], []
        for r in rows[1:]:
            if r[key] is not None:
                gaps.append(top - r["official"])
                costs.append(1 - r[key]["f"])
        gaps, costs = np.array(gaps), np.array(costs)
        ratio = costs / gaps
        out[key] = dict(
            n_crownable=len(costs) + 1,           # + the official leader
            gaps=gaps.tolist(), costs=costs.tolist(),
            ratio_median=float(np.median(ratio)),
            ratio_max=float(ratio.max()),
            ratio_top20_max=float(ratio[:19].max()),
            min_ratio=float(ratio.min()),
            keep_top20_min=float(min(r[key]["f"] for r in rows[1:20]
                                     if r[key] is not None)),
            top20_all=all(r[key] is not None for r in rows[:20]),
        )
        for f in (0.95, 0.9, 0.8, 0.5):
            out[key]["n_at_%d" % round(f * 100)] = 1 + sum(
                r[key] is not None and r[key]["f"] >= f for r in rows[1:])
    second = rows[1]
    out["second"] = dict(model=second["model"],
                         gap=top - second["official"],
                         cost=1 - second["free"]["f"],
                         removed=N_ITEMS[tag] - second["free"]["n_items"])
    out["leader"] = rows[0]["model"]
    out["top20_last"] = rows[19]["model"]
    out["top20_last_gap"] = top - rows[19]["official"]
    return out


def pct(x, d=1):
    return ("%." + str(d) + "f") % (100 * x)


def main():
    lines = []
    def mac(name, val):
        lines.append("\\newcommand{\\%s}{%s}" % (name, val))

    figs = {}
    for tag, pre in (("", "One"), ("rb2_", "Two")):
        path = os.path.join(RES, "crown_%sall.json" % tag)
        if not os.path.exists(path):
            continue
        s = summarize(tag)
        figs[tag] = s
        mac(pre + "Models", s["n_models"])
        mac(pre + "Items", N_ITEMS[tag])
        mac(pre + "SecondGap", pct(s["second"]["gap"]))
        mac(pre + "SecondCost", pct(s["second"]["cost"]))
        mac(pre + "SecondRemoved", s["second"]["removed"])
        mac(pre + "TopTwentyGap", pct(s["top20_last_gap"]))
        for key, K in (("free", "Free"), ("strat", "Strat")):
            v = s[key]
            mac(pre + K + "Crownable", v["n_crownable"])
            mac(pre + K + "RatioMedian", "%.2f" % v["ratio_median"])
            mac(pre + K + "RatioMax", "%.2f" % v["ratio_max"])
            mac(pre + K + "RatioTopTwentyMax", "%.2f" % v["ratio_top20_max"])
            mac(pre + K + "RatioMin", "%.3f" % v["min_ratio"])
            mac(pre + K + "KeepTopTwentyMin", pct(v["keep_top20_min"]))
            for f in (95, 90, 80, 50):
                mac(pre + K + "AtKeep" + {95: "NinetyFive", 90: "Ninety",
                                          80: "Eighty", 50: "Fifty"}[f],
                    v["n_at_%d" % f])

    for tag, pre in (("", "One"), ("rb2_", "Two")):
        path = os.path.join(RES, "benign%s.json" % ("_rb2" if tag else ""))
        if not os.path.exists(path):
            continue
        b = json.load(open(path))
        for r in b["random_sets"]:
            n = {25: "TwentyFive", 50: "Fifty", 100: "Hundred",
                 200: "TwoHundred", 500: "FiveHundred", 1000: "Thousand"}[r["n"]]
            mac(pre + "Keep" + n, pct(r["p_top_kept"], 0))
            mac(pre + "Winners" + n, r["distinct_winners"])
        nat = b["natural"]
        worst = max(nat.items(), key=lambda kv: kv[1]["official_rank"]
                    if kv[1]["n_tied"] == 1 else -1)
        mac(pre + "NatWorstName", worst[0].replace("subset ", "")
            .replace("only ", "").replace("_", "\\_"))
        mac(pre + "NatWorstRank", worst[1]["official_rank"])
        secs = {k: v for k, v in nat.items() if k.startswith("only ")}
        mac(pre + "SectionWinnersDistinct",
            len({v["winner"] for v in secs.values()}))

    # pairwise attainability: items the leader gets right and model j gets
    # wrong must carry at least the gap for the pairwise floor to be exact
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load import load
    d = load()
    S = d["R"] @ d["w0"]
    lead = int(np.argmax(S))
    att = [float(d["w0"][(d["R"][lead] - d["R"][j]) >= 1].sum()) >= S[lead] - S[j]
           for j in range(len(S)) if j != lead]
    mac("OneAttainable", sum(att)); mac("OneTrailing", len(att))

    # table: the leader and the next seven models on RewardBench
    rows = load_rows("")
    top = rows[0]["official"]
    tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "& & & \\multicolumn{2}{c}{Removed to crown} & \\\\",
           "\\cmidrule(lr){4-5}",
           "Model & Score & Gap & Share & Items & Strat.\\\\", "\\midrule"]
    for r in rows[1:8]:
        f, s = r["free"], r["strat"]
        name = r["model"].split("/")[-1].replace("_", "\\_")
        tex.append("%s & %.1f & %.2f & %.2f & %d & %.2f\\\\" % (
            name, 100 * r["official"], 100 * (top - r["official"]), 100 * (1 - f["f"]),
            N_ITEMS[""] - f["n_items"], 100 * (1 - s["f"])))
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "table_top.tex"), "w").write("\n".join(tex) + "\n")

    open(os.path.join(PAPER, "numbers.tex"), "w").write("\n".join(lines) + "\n")

    # benign table: both benchmarks side by side
    rows = []
    for tag in ("", "_rb2"):
        p = os.path.join(RES, "benign%s.json" % tag)
        rows.append({r["n"]: r for r in json.load(open(p))["random_sets"]}
                    if os.path.exists(p) else None)
    tex = ["\\begin{tabular}{rcccc}", "\\toprule",
           "& \\multicolumn{2}{c}{RewardBench} & \\multicolumn{2}{c}{RewardBench 2}\\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
           "$n$ & leader kept & winners & leader kept & winners\\\\", "\\midrule"]
    for n in (25, 50, 100, 200, 500, 1000):
        cells = []
        for r in rows:
            cells += (["%s\\%%" % pct(r[n]["p_top_kept"], 0),
                       str(r[n]["distinct_winners"])] if r else ["--", "--"])
        tex.append("%d & %s\\\\" % (n, " & ".join(cells)))
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "table_benign.tex"), "w").write("\n".join(tex) + "\n")

    # figure: removal needed vs score gap, both benchmarks
    fig, axes = plt.subplots(1, len(figs), figsize=(3.4 * len(figs), 2.5),
                             squeeze=False)
    for ax, (tag, s) in zip(axes[0], figs.items()):
        g = np.array(s["free"]["gaps"]) * 100
        c = np.array(s["free"]["costs"]) * 100
        gs = np.array(s["strat"]["gaps"]) * 100
        cs = np.array(s["strat"]["costs"]) * 100
        lim = max(g.max(), gs.max() if len(gs) else 0) * 1.05
        ax.plot([0, lim], [0, lim], color="0.6", lw=0.8, ls="--",
                label="Prop. 1 bound (y = x)")
        ax.scatter(gs, cs, s=9, color="#C1543A", label="section-preserving",
                   alpha=0.8, lw=0)
        ax.scatter(g, c, s=9, color="#1F3A5F", label="unconstrained", lw=0)
        ax.set_xlabel("official gap to the winner (points)", fontsize=9)
        ax.set_ylabel("share removed to crown (%)", fontsize=9)
        ax.set_title("RewardBench" + (" 2" if tag else ""), fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, None)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "fig_cost.pdf"))
    print(open(os.path.join(PAPER, "numbers.tex")).read())


if __name__ == "__main__":
    main()


# Curated to Win

How much can the choice of validation items move the winner of a reward-model
leaderboard? Analysis of RewardBench and RewardBench 2 public per-item results.

    pip install numpy scipy matplotlib huggingface_hub

    python src/fetch.py            # RewardBench v1 per-item results -> data/compact/
    python src/fetch_rb2.py        # RewardBench 2 -> data/compact_rb2/
    python src/curate.py           # crowning LPs, all models (one to two hours)
    python src/curate.py rb2       # same for RewardBench 2
    python src/benign.py           # random and natural curations
    python src/benign.py rb2       # same for RewardBench 2
    python src/certify.py          # certificates, possible winners, validation sets
    python src/report.py           # paper/numbers.tex, tables, figure
    python src/certify_report.py   # appends certificate macros to numbers.tex, tables

Run `report.py` before `certify_report.py`: the first rewrites `numbers.tex`
and the second appends to it, and the paper needs both.

The paper (`paper/main.tex`) reads every number from `paper/numbers.tex`.

`results/` holds the outputs of `curate.py`, `benign.py` and `certify.py`, so
you can skip those steps to rebuild the paper. `report.py` still needs `data/`
for one check, and `certify.py` checks Prop. 3's hypotheses and the item
certificate by rescoring.

`src/check_official.py` compares rebuilt subset accuracies with RewardBench's
published ones. `src/check_alignment.py` checks item alignment and needs two raw
result files in `data/`, named in its docstring.

## Paper

Adrian Erlikhman. *Curated to Win: Certifying Reward-Model and Judge Selection
Against Evaluation-Set Curation.* ARRSI workshop at AACL-IJCNLP 2026
(accepted; camera-ready in `paper/main.pdf`). Build with `tectonic paper/main.tex`
or pdfLaTeX + BibTeX.

`data/` is not in the repo (about 50 MB of downloaded results); the two fetch
scripts recreate it from `allenai/reward-bench-results` and
`allenai/reward-bench-2-results`.

## Run notes

Each full LP sweep takes one to two hours on a laptop CPU, and the certificates
take seconds. One failed run is worth reporting: the first section-preserving
search started its bisection at a very small kept share, where rounding can
empty a section, and wrongly reported that the winner could not be crowned. The
corrected search scans downward from f = 1.

# Curated to Win

How much can the choice of validation items move the winner of a reward-model
leaderboard? Analysis of RewardBench and RewardBench 2 public per-item results.

    python src/fetch.py            # RewardBench v1 per-item results -> data/compact/
    python src/fetch_rb2.py        # RewardBench 2 -> data/compact_rb2/
    python src/check_alignment.py  # items line up across result files
    python src/curate.py           # crowning LPs, all models  (add rb2 for RB2)
    python src/benign.py           # random and natural curations (add rb2)
    python src/report.py           # paper/numbers.tex, table, figure

The paper (`paper/main.tex`) reads every number from `paper/numbers.tex`.

## Paper

Adrian Erlikhman. *Curated to Win: Certifying Reward-Model and Judge Selection
Against Evaluation-Set Curation.* ARRSI workshop at AACL-IJCNLP 2026
(accepted; camera-ready in `paper/main.pdf`). Build with `tectonic paper/main.tex`
or pdfLaTeX + BibTeX.

`data/` is not in the repo (about 50 MB of downloaded results); the two fetch
scripts recreate it from `allenai/reward-bench-results` and
`allenai/reward-bench-2-results`.

"""Write the OpenReview fields (title, abstract, keywords, TL;DR) as plain
text, with every \\macro in the abstract replaced by its value from
paper/numbers.tex, so the form matches the PDF exactly."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
num = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                      open(os.path.join(ROOT, "paper", "numbers.tex")).read()))
tex = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
title = re.search(r"\\title\{(.*?)\}\n", tex, re.S).group(1).replace("\\\\", " ").strip()
ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).strip()
ab = re.sub(r"\\(\w+)(\{\})?", lambda m: num.get(m.group(1), m.group(0)), ab)
for a, b in (("\\%", "%"), ("$\\times$", "x"), ("~", " "), ("---", "-"), ("--", "-"), ("$", "")):
    ab = ab.replace(a, b)
out = os.path.join(os.path.expanduser("~"), "Downloads", "ARRSI-submission-fields.txt")
open(out, "w", encoding="utf-8").write(
    "TITLE\n%s\n\nABSTRACT\n%s\n\nKEYWORDS\nreward models, LLM-as-a-judge, evaluator selection, certification, "
    "evaluation validity, benchmark curation, RewardBench\n\nTL;DR\nA closed-form certificate says how much of a validation set can be "
    "removed before the selected reward model or judge stops winning; it is exact in items "
    "and tight on RewardBench and RewardBench 2.\n" % (title, ab))
print(open(out, encoding="utf-8").read())

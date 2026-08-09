#!/usr/bin/env python3
"""Turn runs/<id>*_bench.log into rows of benchmark.csv. Deck-generic.

benchmark.py prints the numbers but keeps no file, and re-running it to recover
a number costs a minute per candidate. This parses the saved logs so the CSV is
derived from the evidence rather than retyped from it.

Repeated runs of one candidate (repeat.sh writes <id>.rep<k>_bench.log, ab.sh
writes <id>.ab<k>_bench.log) are grouped by the id before the first dot and
reduced to a per-shape **median over reps**, with min/max kept so the spread
stays visible. That spread is not cosmetic: re-running task 01's c2 unchanged
moved T=1 by 37%, so a single run cannot settle a promotion. The geomean row is
recomputed from the median peak fractions rather than copied from any one run's
`peak_fraction:` line.

Why this is generic where 01's hand-written version was not: the four problems
print different metrics. 01 and 02 report tflops/gbps/ms plus a separate
`shape=N solution_peak_fraction=` line; 03 reports tok_s/decode_wall_s and
carries peak_fraction inline; 04 reports sps/wall_s inline. Rather than hardcode
one shape vocabulary, this harvests every `key=value` on a solution line and
reduces each key independently. Add a metric to benchmark.py and it appears
here with no edit.

Optional `shape_labels.json` in the task dir maps shape index -> label
(e.g. {"0": "T2048"}), purely for readability.

    ../swarm/harness/collect.py runs/*_bench.log > benchmark.csv
    ../swarm/harness/collect.py runs/ab_c8_vs_c10/*_bench.log

Lives in the harness, not in a task's tools/, because it is genuinely shared:
four tasks in trial 2 each received a copy and all four left it byte-identical.
A tool nobody needs to change does not belong in the directory reserved for the
tools a task writes itself.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from math import exp, log
from pathlib import Path
from statistics import median

# Anchored to line start on purpose. benchmark.py also emits a structured event
# stream — `benchmark_event event=variant_end shape=0 variant=solution
# ts=2026-...` — which contains the same shape/variant keys plus an ISO
# timestamp that is not a number. Matching those would double-count every shape
# and crash on `ts=`. Only the human-readable summary line starts with `shape=`.
SOLUTION = re.compile(r"^shape=(\d+)\s+variant=solution\s+(.*)$", re.M)
KV = re.compile(r"([a-z_]+)=([-\d.eE+]+)")
FRAC_LINE = re.compile(r"^shape=(\d+)\s+solution_peak_fraction=([\d.eE+-]+)", re.M)

# id -> shape_idx -> metric -> [value per rep]
runs: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
)

paths = [Path(p) for p in sys.argv[1:]]
if not paths:
    sys.exit(__doc__)

labels = {}
for parent in {p.parent for p in paths} | {p.parent.parent for p in paths}:
    f = parent / "shape_labels.json"
    if f.is_file():
        labels = {int(k): v for k, v in json.loads(f.read_text()).items()}
        break

for path in sorted(paths):
    cid = path.name.removesuffix("_bench.log").split(".")[0]   # c2.rep3 -> c2
    text = path.read_text()

    for m in SOLUTION.finditer(text):
        idx = int(m[1])
        for key, val in KV.findall(m[2]):
            try:
                runs[cid][idx][key].append(float(val))
            except ValueError:
                pass    # non-numeric field on a metrics line; ignore rather than die

    # 01/02 emit the fraction on its own line instead of inline.
    for m in FRAC_LINE.finditer(text):
        runs[cid][int(m[1])]["peak_fraction"].append(float(m[2]))

metrics = sorted({k for c in runs.values() for s in c.values() for k in s})
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(
    ["candidate", "shape_idx", "label", "reps"]
    + [f"{m}_{stat}" for m in metrics for stat in ("median", "min", "max")]
)

for cid in sorted(runs, key=lambda c: (len(c), c)):
    med_fracs = []
    for idx in sorted(runs[cid]):
        shape = runs[cid][idx]
        row = [cid, idx, labels.get(idx, ""), max(len(v) for v in shape.values())]
        for m in metrics:
            vals = shape.get(m)
            row += ([f"{median(vals):.4f}", f"{min(vals):.4f}", f"{max(vals):.4f}"]
                    if vals else ["", "", ""])
        w.writerow(row)
        if "peak_fraction" in shape:
            med_fracs.append(median(shape["peak_fraction"]))

    if med_fracs:
        gmean = exp(sum(log(max(f, 1e-9)) for f in med_fracs) / len(med_fracs))
        nreps = max(len(v) for s in runs[cid].values() for v in s.values())
        row = [cid, "geomean", "", nreps] + [""] * (3 * len(metrics))
        # Land the geomean under peak_fraction_median, so the score sits in the
        # same column as the per-shape fractions it was computed from.
        row[4 + 3 * metrics.index("peak_fraction")] = f"{gmean:.4f}"
        w.writerow(row)

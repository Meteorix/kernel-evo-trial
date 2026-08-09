# 03-megaqwen-decode — log

**APPEND-ONLY.** Never edit or delete an entry, including your own from an earlier
cycle in the same session. A wrong call that was later corrected is the most useful
thing in this file; deleting it leaves a record that looks like it was right all along.

Append with the harness script — it stamps the time and the commit, and refuses an
entry that is missing a hypothesis, a result, or notes:

```bash
../status.sh <NN> <id> <verdict> [--score S] [--replicated] <<'EOF'
**Hypothesis** — what was believed, and what would falsify it. Written BEFORE the
measurement wherever the order allows.
**Result** — what came back. The number, and whether it cleared the noise floor.
**Notes** — what a reviewer needs that the number does not carry: the control arm,
the mechanism, what you now think is wrong, what you did not get to.
EOF
```

`<verdict>` is one of `promoted` `rejected` `baseline` `note` `selftest` `blocked`,
so the log greps into a results table.

**There is no champion field.** The champion is the last `promoted` entry. A separate
summary line drifts from the log the first time a cycle is interrupted, and then reads
as authoritative while being stale.

**Order of operations:** commit the work, *then* append, *then* commit this file. The
entry records `HEAD` at append time, so that sequence makes the commit id point at the
work being described rather than at the note describing it.

## What belongs in an entry

- **Every candidate**, including rejected ones and control arms. A rejection with its
  reasoning is worth more than a promotion without one — it closes a branch of the
  search for whoever comes next.
- **Every measurement of the instrument** — null A/B results, noise floors,
  replications. Playbook rules 14 and 15.
- **Anything you now believe is wrong**, including in an earlier entry of your own.
  Append the correction; do not edit the original.
- **What you did not finish.** A cycle that ran out of time and said so is reviewable;
  one that quietly stopped is not.

## Where the other files stand

`CONTRACT.md` — the transcribed task plus the hand-counted work model (rule 2).
`HYPOTHESES.md` — the open backlog, which is *mutable*: hypotheses open and close.
`candidates.jsonl` — the machine-readable ledger, one row per candidate.
This file — the human-readable history, in order, that explains why the ledger looks
the way it does.

---

<!-- Entries are appended below this line by ../status.sh. Nothing above it changes. -->

### bootstrap · `note`

Seeded by `start-trial.sh`. No contract, no candidates.

**Hypothesis** — none yet. Playbook rule 10 fixes the order before any candidate:
harness → reference validated against a slow, obviously correct ground truth →
broken-kernel self-test asserted FAIL → first candidate.
**Result** — nothing measured.
**Notes** — **there is no `CONTRACT.md`, and writing one is the first job.** Transcribe
it from `kernelbench.com/benchmarks/cuda/problems-rtx2070/03_*/` — `PROMPT.txt`, `problem.yaml`,
`shapes.py`, `check.py`, `benchmark.py` — inventing nothing (rule 1). Then rule 2: the
hand-counted work model, **before** the first candidate — executed FLOPs and bytes per
shape, arithmetic intensity, which roof binds, the floor in ms. Do not take the numerator
from `problem.yaml`; on this deck its formulas were wrong in every way available.

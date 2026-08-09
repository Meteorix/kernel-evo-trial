# 02-deepseek-nsa — log

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
`candidates.jsonl` — KDA's machine-readable ledger, one row per candidate. You create it.
This file — the human-readable history, in order, that explains why the ledger looks
the way it does.

---

<!-- Entries are appended below this line by ../status.sh. Nothing above it changes. -->

### bootstrap · `note`

Seeded by `start-trial.sh`. Nothing here is measured yet.

**What you were given** — this log and the three documents beside it: `prompt.md` (what
the task is), `KDA.md` (how to optimise a kernel), `SWARM.md` (how to work alongside
three other workers). Four files, no directories. That is all, deliberately.

Everything else is named by one of those three, and you create it: `CONTRACT.md`,
`candidates.jsonl`, `candidates/`, `tools/` and `profile/` come from KDA; this log comes
from SWARM.

**What you must write** — everything else, in playbook rule 10's order:

1. `CONTRACT.md` — transcribe from `kernelbench.com/benchmarks/cuda/problems-rtx2070/02_*/`
   (`PROMPT.txt`, `problem.yaml`, `shapes.py`, `check.py`, `benchmark.py`),
   inventing nothing (rule 1).
2. The **hand-counted work model** into `CONTRACT.md`, before any candidate (rule 2) —
   executed FLOPs and bytes per shape, arithmetic intensity, which roof binds, the floor
   in ms. Do not take the numerator from `problem.yaml`; on this deck its formulas were
   wrong in every way available.
3. A **ground truth** the reference is validated against — slow, obvious, independent.
4. A **broken kernel** that `check.py` is shown to FAIL (rule 4), and a `smoke.py`
   covering the shapes `check.py` does not.
5. Only then, the first candidate.

**No `prof_driver.py` is provided, and that is on purpose.** The shared one used to
select a shape with problem 03's knob; three tasks rewrote it and the fourth silently
profiled the wrong shape for a whole cycle. Write your own, validate the selector against
the real graded list, and `sys.exit` on a miss — playbook rule 28.

**Hypothesis** — none yet.
**Result** — nothing measured.
**Notes** — no contract, no candidates, no measurements. Start at 1.

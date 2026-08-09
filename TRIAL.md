# Trial 2026-08-08-t3

Started 2026-08-08T23:03:40-07:00. A trial does not end: a leader session resumes it and keeps
optimising. Each task's `STATUS.md` is an append-only log, which is what makes that
possible.

## Pinned inputs

| | |
| --- | --- |
| tasks | 01 02 03 04 |
| `KDA.md`, `swarm/` | **copied** from the rig at `511b741` — method and apparatus |
| `kernelbench.com` | `824a699` — **the comparison instrument** |
| contracts | none shipped; the trial writes its own (rule 1, then rule 2) |

`KDA.md` and `swarm/` are **copies**, not submodules: the trial is portable, and it can
patch its own harness without changing the starting line for any other trial. A controlled
comparison is still available — check the rig out at an older SHA and start a trial from
there, and the knowledge variable is held fixed. Record here if you deliberately started
from an older rig.

## What this trial is testing

<!-- Fill this in before the first cycle. "Can it get faster" is not a hypothesis.
     Pre-register predictions here, with what would falsify them. A prediction written
     after the measurement is a story. -->

## How results are read

Rank on **milliseconds against the hand-counted floor**, not `peak_fraction`. Every known
defect in the bench lives in a denominator — see ../plan.md §2.2 and `KERNELBENCH.md`.

## Log

<!-- Pre-registrations and results accumulate here. Since the trial does not end, this is
     a register rather than a paper: append, never rewrite. -->

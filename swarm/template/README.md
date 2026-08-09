# Task scaffolding

What `start-trial.sh` seeds into every task directory, and — more importantly — what it
deliberately does not.

```
<NN>-<name>/
  STATUS.md          append-only log, seeded with a bootstrap entry
  HYPOTHESES.md      the backlog, seeded empty
  candidates.jsonl   the ledger, seeded empty
  config.json        per-task dials for this trial
  tools/collect.py   parses bench output into the ledger
  candidates/
```

Everything else a task needs — `CONTRACT.md`, the work model, a ground truth, a broken
kernel, `smoke.py`, a profiling driver, every probe — the task writes itself. `runs/` and
`profile/` are created by the harness that writes into them.

## Why the seed is this small

Measured across trial t2's four tasks, which all started from a much larger seed:

| seeded | what happened |
| --- | --- |
| `collect.py` | **kept unchanged by 4/4** |
| `HYPOTHESES.md` | grew 36 → 223–314 lines in 4/4 |
| `smoke.py` | **rewritten by 4/4** |
| `docs/draft.md`, `docs/report.md` | **untouched in 8/8** |
| `prof_driver.py` | rewritten by 3/4 — and the fourth is the problem |

The `prof_driver.py` row is the argument for cutting rather than keeping. Its shape
selector used problem 03's own knob, so on every other problem it profiled the *default*
shape whatever it was asked for, while the report looked entirely normal. Three tasks
rewrote it and never noticed. The one task that kept it lost a whole cycle of profiles
(playbook rule 28).

So a stub is not free. It is a **plausible default that gets adopted without scrutiny**,
and the only case where it survived unedited is the case where it did damage. Seed what is
demonstrably generic; make everything else the task's own work, where it gets read.

The shared pieces that remain — the ledger schema, the STATUS log format, `collect.py` —
exist because the leader has to compare four tasks. That is the whole justification, and it
does not extend to anything a task would rewrite anyway.

## The layout, enforced by convention not by code

**Every script a task writes goes in `tools/`.** A probe left at top level is invisible to
the next worker, which then rebuilds it. Every candidate goes in `candidates/` and is
immutable once benched — a candidate edited after measurement makes its ledger row a lie.

`STATUS.md` is **append-only**: entries are added with `../swarm/harness/status.sh`, never
edited. It is what makes a trial resumable, and a rewritten dashboard is not.

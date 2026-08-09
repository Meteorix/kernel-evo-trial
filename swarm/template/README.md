# Task scaffolding

What `start-trial.sh` seeds into every task directory, and — more importantly — what it
deliberately does not.

```
<NN>-<name>/
  prompt.md          what the task is — the bench's PROMPT.txt, verbatim
  KDA.md    ->       how to optimise a kernel        (symlink to the trial root)
  SWARM.md  ->       how to work with three others   (symlink to the trial root)
  STATUS.md          append-only log, seeded with a bootstrap entry
  candidates.jsonl   the ledger, seeded empty
```

Five files, no directories: three inputs, two outputs. Everything else a task needs — `CONTRACT.md`, the work model,
a ground truth, a broken kernel, `smoke.py`, a profiling driver, every probe — the task
writes itself. `candidates/`, `runs/` and `profile/` are created by whoever first writes
into them; git does not track empty directories, so seeding them achieves nothing.

## Why there are two records and not one

`candidates.jsonl` and `STATUS.md` both have a row per candidate, which looks like double
entry. It is not, and the split is worth keeping: **the ledger is what a machine reads
across tasks** — per-shape milliseconds, register counts, reps, ambient load, in fixed
fields the leader can compare four tasks on — while **`STATUS.md` is what a person reads
within one**, in prose, including the reasoning a number cannot carry. Neither substitutes.
Hypotheses are entries in the log, not a separate file. The obvious objection — that an
append-only log cannot answer *what is still open* — is answered by deriving it:
`status.sh --open` lists ids opened with a `hypothesis` entry and never followed by a
`closed` one. That is strictly better than the hand-kept `## Open` section it replaces,
which was an assertion nothing forced to stay true, and it keeps re-rankings visible
instead of overwriting them.

## What was dropped, and why

`collect.py` moved to `swarm/harness/`. Four tasks each got a copy and all four left it
byte-identical, which is the definition of a shared tool: it does not belong in the
directory reserved for what a task writes itself.

`config.json` is gone. Trial 1 filled one in per task with stall thresholds and measurement
dials; **nothing ever read it** — no harness script, no tool — and trial 2's four tasks all
left it at its 60-byte stub. Policy that nothing enforces is worse than policy in prose,
because it reads as binding. Stall budgets belong in the leader's `SKILL.md`, where the
leader actually looks. `cycles.jsonl` went the same way: 7–11 rows per task in t1, 0–2 in
t2.

## Why the seed is this small

Measured across trial t2's four tasks, which all started from a much larger seed:

| seeded | what happened |
| --- | --- |
| `collect.py` | **kept unchanged by 4/4** — so it moved to `harness/`, where shared things live |
| `HYPOTHESES.md` | grew 36 → 223–314 lines in 4/4 — kept, but folded into the log |
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

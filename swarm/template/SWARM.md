# Working in the swarm

You are one worker on one task. Three other workers are doing the same thing on other
tasks, on **one GPU**, with a leader merging your work. This file is the whole protocol —
your cycle brief adds only what is specific to this cycle.

## What you own, and what you must not touch

**Yours:** your own task directory. Commit only there.

**Never edit, on any pretext:**

- **Anything under `kernelbench.com/`.** `check.py`, `benchmark.py`, `shapes.py`,
  `reference.py`, `problem.yaml` and `PROMPT.txt` are bench-rule-forbidden *and* shared by
  every worker: an edit by you silently invalidates every number the other three produce.
  `solution.py` is the one exception — it is the install slot the harness writes.
  `gpu.sh` aborts every run if the bench is dirty, so this fails loudly, not quietly.
- **`KDA.md`, `SWARM.md`, `swarm/`.** Leader-owned. If you want one changed, say so in your
  report with the evidence; a rule that arrives with a measurement behind it gets promoted.
- **Any other task's directory.**

Do not disable the numeric-stress gate (`KBH_NUMERIC_STRESS=0`). Do not run `find /` — the
root is a network mount and it takes about ten minutes.

## The GPU is serialised, and it is not the bottleneck

Every GPU-touching command goes through the lease wrappers — `run.sh`, `ab.sh`,
`repeat.sh`, `profile.sh`, all of which call `gpu.sh`. Never invoke `check.py`,
`benchmark.py` or `ncu` directly: an unleased run corrupts whoever holds the lease, and
their measurement is the one that matters.

Waiting on the lease is normal and costs you nothing, because **the card is idle most of
the time** — measured at ~30% occupancy with four workers active. The binding constraint is
thinking and writing, not silicon. So:

- **Do everything you can off the lease.** `nvcc -Xptxas -v -arch=sm_75 -cubin` for
  registers, shared memory, spills and stack frame needs no GPU and has killed candidates in
  seconds. So does `cuobjdump -sass`, and re-reading a captured profile with
  `ncu --import`. The work model, the hypotheses and the ledger are all free.
- **Do not hold the lease while you think.** Take it, measure, release.

## What the harness gives you

`../swarm/harness/` holds everything shared: `run.sh`, `ab.sh`, `repeat.sh`, `profile.sh`,
`gpu.sh`, and `collect.py`, which turns saved bench logs into a per-shape CSV with the
median over reps and the spread kept visible.

```bash
../swarm/harness/collect.py runs/*_bench.log > benchmark.csv
```

**Your `tools/` is for tools you write** — probes, ground truths, work models, a profiling
driver. Put every script there rather than at the top level, or the next worker cannot find
it and rebuilds it. Nothing is seeded there, deliberately: the shared stub that used to be
handed out selected the wrong shape on three of four problems, and the one task that
trusted it lost a cycle of profiles.

## The rules that decide whether your number is believed

- **Write the hypothesis before the measurement.** A prediction recorded afterwards is a
  story. Name what would falsify it, and name the null arm — the variant you expect to be
  worth *nothing*. A confirmed null is stronger evidence than a confirmed win.
- **Measure your own instrument first.** Run a candidate against a byte-identical copy of
  itself. Whatever spread that shows is your noise floor, and any delta under it is not a
  result. One task found a null arm reading **+117%** on its smallest shape — nothing on
  that shape was measurable at all, including the candidate built for it.
- **Replicate before promoting.** One paired A/B is one draw.
- **Use a control candidate when a change has more than one part.** Isolating the part you
  pre-registered is the only way to know the mechanism was the mechanism; without it a
  ledger records a confirmed cause that was not the cause.
- **Rank on milliseconds against your own hand-counted floor**, not `peak_fraction`. Every
  known defect in this bench lives in a denominator.

## Hypotheses live in the log

There is no `HYPOTHESES.md`. Open a question as an entry with the `hypothesis` verdict and
an id of your choosing (`H1`, `H-lcg`), settle it with a `closed` entry using the same id:

```bash
../swarm/harness/status.sh <NN> H1 hypothesis <<'EOF'
**Hypothesis** — env tiling is the whole game; a block must own >= 15 envs.
**Result** — not yet measured.
**Notes** — sized from the L2 model; falsified if env_tile 16 -> 32 pays.
EOF

../swarm/harness/status.sh --open <NN>     # what is still open, derived
```

**The open set is derived, never maintained.** A hand-kept "## Open" section is an
assertion, and nothing forces it to match what actually happened; an id opened and not
later closed *is* open, by construction. Re-sizing a hypothesis is a new entry, not an
edit — so the record shows that you re-ranked it and why, which an overwritten list
destroys.

## The log is the deliverable

`STATUS.md` is **append-only**. Add entries with `../swarm/harness/status.sh`; never edit
or delete one, including your own from an earlier cycle. A wrong call you later corrected is
the most useful thing in the file — append the correction, keep the original.

Every candidate gets an entry, **including rejects and control arms**. A rejection with its
reasoning closes a branch of the search for whoever comes next, which is worth more than an
unexplained promotion. Nulls, noise floors and replications go in too.

This is also what makes the trial resumable: a leader session picks up from your log, so a
cycle that ran out of time and *said so* is worth more than one that quietly stopped.

## Reporting, merging, and being interrupted

End your cycle by updating `STATUS.md` and `candidates.jsonl`, committing
to your branch, and reporting: what you tested, what you promoted or rejected and on what
evidence, and what the next cycle should do first. **State plainly what you could not
finish.**

The leader runs a merge gate and is not a rubber stamp. It checks that your commits stay in
your task directory, that the bench is clean, that `STATUS.md` was appended to rather than
rewritten, and that a promoted candidate has a `PASS`, enough reps, recorded ambient load,
and a named hypothesis. It may re-run your numbers.

You can be stopped mid-cycle. Commit early and often; uncommitted work is the only thing
that gets lost.

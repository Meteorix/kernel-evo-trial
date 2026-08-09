---
name: evolve
description: Run one round of the cuda-bench self-evolving kernel loop — read state, detect stalls, spawn per-task workers, gate and merge their commits, accumulate lessons, rewrite STATUS.md. Use when the user asks to start, continue, or run the kernel evolution loop, or invokes /evolve (usually via /loop /evolve for unattended operation).
---

# evolve — one round of the cuda-bench self-evolving loop

You are the **leader** of one **trial**, which lives in
`../trials/<name>/`. Everything the trial produces stays there.

`` is the **frozen starting point** — harness, template, and
the bench's transcribed contracts. A trial never writes to it; fixes are promoted
back deliberately in their own commit. Start a trial with
`start-trial.sh <name>`.

Full design: `../kda/PLAYBOOK.md`'s companion at the repo root, `PLAN.md`
(the method, shared by all trials).
Dials: `<trial>/evolve.config.json`. Read both if anything here is
ambiguous — this file is the procedure, the plan is the reasoning.

**There is no round.** Scheduling is asynchronous and per-task: each task's next
cycle starts when *that* task's worker reports, not when the slowest of four does.
One invocation of this skill = one pass of the handler below. Under `/loop` you
will be re-invoked, and worker completions will also wake you directly.

Work from `/home/meteorix/proj` on `master`; the trial directory is your root. You own `master` and every shared
file. Workers own only their task directories.

## 0a. Report to the user every 5 minutes

**Standing requirement, not a default you may tune away.** While any worker is in
flight, post a short status to the user at least every 5 minutes, and schedule the
next wake-up with `ScheduleWakeup(delaySeconds: 300)` carrying the same `/loop` prompt.

This deliberately contradicts the usual advice against short polling for
harness-tracked work. The user asked for it, and the reason it is worth the tokens is
that a trial is hours long and otherwise silent: worker completions arrive
unpredictably, and between them there is no signal at all unless the leader produces
one.

Each report should carry, briefly:

- what `./wstat.sh` says — per worktree: BUSY / QUEUED / working / idle / STUCK?, plus
  the lease holder and the last few lease-log lines
- what changed since the last report (commits on worker branches, merges, promotions)
- anything that needs the user

**Do not infer liveness from file mtimes.** A worker that has stopped writing and
started running looks identical to a dead one; this produced two wrong calls in trial
1, in opposite directions. `wstat.sh` uses the process table and the lease log, and it
separates BUSY (holding the lease, working legitimately) from QUEUED (blocked in
`flock`, which is normal) from STUCK? (neither, past the threshold) — three states that
look the same if you only measure elapsed time.

If nothing has changed since the last report, say exactly that in one line. A quiet
report is information; silence is not.

## 0b. Draw the trace when the user asks how the trial is going

`./timeline.py` renders the whole trial as one page — four lanes on a shared time
axis, GPU holds and queue blocks, commit ticks, merge rules, cycle bands — from data
already on disk (the lease audit log plus per-branch `git log`). Publish it with
`Artifact`; re-running against the same file path updates the same URL.

```bash
./timeline.py                 # writes ./timeline.html, prints a one-line summary
./timeline.py --out /tmp/t.html
```

Use it when the user asks for a timeline, a trace, or "how is it going" in a way that
wants shape rather than a status line. `wstat.sh` answers *right now*; `STATUS.md`
answers *what a worker concluded*; this answers **where the time actually went**, and
that question has no other artefact.

It is worth running unprompted at least once per trial, because the reading is
usually a surprise. In t2 it showed the GPU idle ~70% of the wall clock — four agents
contending for one serialized lease still could not keep the card busy, which means
the card was never the binding constraint the parallel design assumed. It also
localised nearly all queueing to a *single* long lease rather than to steady
contention. Neither fact was visible in any per-cycle report.

Read the occupancy figure as an **upper bound**: a lease covers process start,
compile and teardown, not just kernel time.

## 0. Preconditions

Abort the round and say why if:

- `git -C kernelbench.com status --porcelain` is non-empty. The bench is shared
  by all four workers; a dirty tree means every task's numbers are suspect.
  **This halts the whole loop, not one task** (config `dirty_bench_halts_everything`).
- `master` has uncommitted changes you did not make this round.

## 1. Read state

Read all of it before deciding anything:

- `<NN>-*/STATUS.md`, `<NN>-*/CONTRACT.md`, `<NN>-*/HYPOTHESES.md`,
  `<NN>-*/cycles.jsonl`, `<NN>-*/config.json` — each task owns its own. These are
  authoritative.
- `STATUS.md` — a roll-up only. If it disagrees with a task's
  own file, the task's file wins.
- the tail of each `<NN>-*/candidates.jsonl`
- `/tmp/cuda-bench.gpu.log` — lease waits and hold durations
- `git log --oneline -15`

## 2. Detect stalls

Per task, from `<task>/cycles.jsonl`. **Progress means frontier movement, not a
promotion.** A round counts as progress if any of:

- the champion improved, **or**
- a hypothesis was closed with evidence, **or**
- new hypotheses were opened from evidence (a profile, a SASS read, an
  arithmetic floor), **or**
- a `ramping` milestone was passed (harness → reference validated → self-test
  FAILing)

Task 01's `c7` (−16%) and `c9` (−15%) were both rejected and both among the most
valuable rounds in its ledger. **Never treat a rejection as a stall.**

Counters reset on progress. Compare against `<task>/config.json` → `stall`. Deck-level budget and
scheduling live in `evolve.config.json`; per-task dials do not.

Classify:

| Class | Signature | Response |
| --- | --- | --- |
| Barren | candidates measure cleanly, nothing opens or closes | soft → hard escalation |
| Broken | 3 consecutive build or `check.py` failures | escalate early — mechanics, not ideas |
| Thrash | same hypothesis re-tested under new ids; champion oscillating | escalate |
| Exhausted | backlog empty, or all open entries predict below the noise floor | **not a stall** — mark `converged`, report as success |
| Blocked | ambiguity only the user can settle | escalate immediately, consume no budget |

At **50% of budget**, intervene yourself before escalating, in this order:
1. **Force a profile.** No task may reach `stalled` without an `ncu` run on its
   current champion. Three of task 01's promotions trace to a profile.
2. Recheck the arithmetic floor — task 01's T=1 frontier was closed by
   arithmetic, not by a candidate.
3. Re-frame with a fresh worker and an explicitly different angle.

At **100%**, set the task `stalled`, stop scheduling it, and write the escalation
(§6).

## 3. Schedule — per task, no barrier

For **each** task independently: if it has no worker in flight and is not
`converged`/`stalled`/`blocked`, spawn its next cycle now. Do not wait for other
tasks. `frontier-thin` tasks get a worker only every
`frontier_thin_every_n_cycles`.

**Respawn backoff (the guard the barrier used to provide for free).** A cycle that
ended in under `worker.min_cycle_seconds` *without progress* is a fast failure:
back off, and after `worker.fast_fail_backoff` consecutive ones stop scheduling
that task and escalate. An agent that errors out in twenty seconds would otherwise
respawn forever at the speed of its own failure.

**Liveness, every pass.** Check the process table by worktree and the lease log for
any single call or hold past `stall.single_call_minutes`. A worker stuck inside one
tool call is invisible to a detector that only looks at cycle boundaries — this
cost round 1 ten minutes. Do NOT infer liveness from file mtimes: a worker that has
stopped writing and started running looks identical to a dead one. Use the process
table (with cwd per worktree) and `/tmp/cuda-bench.gpu.log`.

## 4a. `STATUS.md` is an append-only log

Each task's `STATUS.md` is a chronological record, one entry per iteration, appended
with `./status.sh` — which stamps the timestamp and the commit, and rejects an entry
missing a **Hypothesis**, a **Result**, or **Notes**.

```bash
./status.sh <NN> <id> <verdict> [--score S] [--replicated] <<'EOF'
**Hypothesis** — ...
**Result** — ...
**Notes** — ...
EOF
```

Order matters: **commit the work, append, then commit the status change.** The entry
records `HEAD` at append time, so that sequence points the commit id at the work
rather than at the note about it.

**It is not a dashboard, and there is no champion field** — the champion is the last
`promoted` entry. This is the correction to what t2 ran with. There, every task
invented its own layout and rewrote it each cycle, which produced two failures at
once: four tasks could not be reviewed the same way, and a killed cycle left a stale
summary that still read as authoritative (task 04's header claimed cycle 1 after it
had committed a cycle-2 sweep). A log cannot go stale, because nothing in it claims
to be current.

Every candidate gets an entry, **including rejects and control arms** — a rejection
with its reasoning closes a branch of the search for the next worker, which is worth
more than an unexplained promotion. So do null A/Bs, noise floors and replications
(rules 14, 15), and any belief you now think was wrong: append the correction, never
edit the original.

Put this in every worker brief, and check it at the merge gate (§5).

## 4. Spawn workers

One `Agent` call per scheduled task, **all in a single message** so they run
concurrently. Workers are per-round: one cycle, then exit. Brief each with:

```
You are worker-<NN> for cuda-bench task <NN>-<name>. ONE cycle, then report and exit.

Worktree: /home/meteorix/proj-wt/cb<NN>   (work here, not in /home/meteorix/proj)
Branch:   task/<NN>-<slug>
Task dir: <worktree>/../trials/<trial>/<NN>-<name>/

READ FIRST: ./CONTRACT.md (yours, self-contained), ./HYPOTHESES.md, ./STATUS.md,
  ./config.json, the tail of ./candidates.jsonl, and ../../../../kda/PLAYBOOK.md.
  Your task directory is self-contained — you should not need another task's files.

Your hypotheses this round, highest value first: <from HYPOTHESES.md>
Closed — do NOT re-spend candidates on these: <closed list>

THE GPU IS SHARED AND SERIAL. One RTX 2070, up to four workers.
- Every GPU command goes through ../gpu.sh. Never invoke check.py, benchmark.py
  or ncu directly.
- ../run.sh check|bench, ../repeat.sh, ../ab.sh and ../profile.sh already take
  the lease. ../run.sh prebuild deliberately does not — it compiles with no
  CUDA context, so run it before queueing.
- A candidate does not enter the GPU queue until it has (a) a written hypothesis
  and (b) an `nvcc -Xptxas -v` result. Playbook rule 6. `-Xptxas -v`,
  `cuobjdump -sass` and `ncu --import` on an existing report need no GPU and no
  lease — do all of that first. `-Xptxas -v` killed candidate c9 in seconds.
- Exit 75 from gpu.sh means the lease is busy: go do CPU work and retry.

CAPTURED PROFILES LIVE IN THE LEADER CHECKOUT, NOT YOUR WORKTREE.
`.gitignore` has `*/profile/*/*.ncu-rep` (several MB each); only the hand-written
REPORT.md beside each one is tracked. Your worktree therefore has the directory
structure and the REPORT.md files but none of the binaries. Read existing
profiles read-only from the absolute leader path:
  /home/meteorix/proj/../trials/<trial>/<NN>-<name>/profile/<run>/report.ncu-rep
`ncu --import` on those needs no GPU and no lease. If you capture a NEW profile,
commit its REPORT.md as usual and name the run directory in your report — the
leader copies the binary into the shared archive at merge time.

Do NOT run `find /`. /mnt/c is a 9p mount of the Windows filesystem and a full
scan takes minutes. Scope searches to /home/meteorix/proj and /home/meteorix/proj-wt.

MEASURE HONESTLY.
- Promotion decisions use ../ab.sh (paired, interleaved, one lease hold), >= 3
  reps, and are read on ms_min as well as the median. Task 01's T=512 is bimodal
  across a 1.64x clock-bin gap; a median over reps reports which bin came up
  more often. Playbook rule 5.
- Record `ambient` (loadavg, workers active) in the ledger — other workers were
  compiling while you measured.
- One variable per candidate. If hardware forces two, record it as a caveat.

BOUNDARIES.
- YOUR TASK DIRECTORY IS SELF-CONTAINED and has a fixed shape: six state files
  (CONTRACT.md, STATUS.md, HYPOTHESES.md, candidates.jsonl, cycles.jsonl,
  config.json) and five directories (tools/, candidates/, docs/, runs/, profile/).
  All of it is YOURS — edit freely. You should never need another task's directory.
- PUT EVERY SCRIPT YOU WRITE IN tools/. Probes and one-off models accumulate fast;
  one task hit 30 top-level entries, 20 of them ad-hoc probes. Keep them — they are
  how a later cycle re-checks a claim — but keep them out of the way of the six
  files the next worker reads first. A tool that locates itself sits one level down:
  Path(__file__).resolve().parent.parent is the task dir.
- Leader-owned, ask do not edit: ../gpu.sh, ../run.sh, ../ab.sh, ../repeat.sh,
  ../profile.sh, ../template/, ../STATUS.md (a roll-up only), ../evolve.config.json
  and ../../../../kda/PLAYBOOK.md. These are shared BECAUSE sharing is the point: six
  harness bugs found by one task were fixed for all four in a day. Report bugs in
  them; do not fork them.
- NEVER edit anything under kernelbench.com. check.py, benchmark.py, shapes.py,
  reference.py, problem.yaml and PROMPT.txt are bench-rule-forbidden AND shared
  with three other workers; editing one invalidates everybody's numbers.
- `git rebase master` before committing.

WRITE IT DOWN OR IT DID NOT HAPPEN. You are ephemeral; the files are not.
Before exiting: append to candidates.jsonl (promoted AND rejected, each with a
`note` saying WHY), update HYPOTHESES.md (open and closed), commit.

If you hit an ambiguity that affects CORRECTNESS, stop and report it — do not
assume. If it affects only performance, pick the defensible reading, record the
assumption, and continue.

REPORT: candidate id, hypothesis tested, outcome with numbers, hypotheses
opened/closed, commit sha, GPU minutes used, anything the leader should
propagate to other tasks.
```

## 5. Merge gate — you are not a rubber stamp

Per worker branch, before merging a promotion:

- `candidates.jsonl` entry has `check: PASS`, `framework` ≠ `pytorch_only`,
  `reps >= 3`, `ms_min` present, `ambient` recorded, and a `note` explaining why
- the candidate names the hypothesis it tested, and that hypothesis is opened or
  closed in `HYPOTHESES.md` either way
- `git diff --stat master..task/<NN>-*` touches nothing outside the task dir
- `git -C kernelbench.com status --porcelain` clean
- **`STATUS.md` was appended to, not rewritten.** It is an append-only log (§4a), so
  the diff must be pure addition:

  ```bash
  git diff master...<branch> -- '*/STATUS.md' | grep '^-[^-]' && echo "REWRITTEN — reject"
  ```

  A worker that rewrote it destroyed the history the file exists for. Ask it to
  restore the deleted entries and append its correction instead; do not fix it
  yourself, because the worker is the only one who knows what it meant to say.
- **the cycle's work has an entry.** A branch carrying candidates but no new
  `STATUS.md` entry is not reviewable, and the ledger alone does not substitute —
  `candidates.jsonl` records what was measured, not what was believed beforehand.
- **you re-run `check.py` and at least one bench rep yourself.** A number only
  the worker that produced it has ever seen is the failure that costs the whole
  run its credibility. Budget lease time for this.

Then `git merge --no-ff` one branch at a time.

**Do NOT `git reset --hard` a worktree after merging.** A worker that has reported
is not necessarily dead — sending it a message revives it, and it may commit again.
Resetting its worktree while it is live, or while its successor is already working
there, risks destroying work or putting two agents in one directory. Workers rebase
onto master themselves (their brief says so); let them. Only reset a worktree you
have confirmed is idle *and* has no successor spawned.

Corollary: **once you have spawned a task's next cycle, stop messaging the previous
worker.** If a correction needs to reach that task, put it in the new worker's brief
instead.

**Archive any new profile binaries.** `.ncu-rep` files are gitignored, so a
profile a worker captured stays in its worktree and is invisible to every future
worker. For each run directory named in a worker's report, copy it into the
leader checkout:

```bash
cp /home/meteorix/proj-wt/cb<NN>/<task>/profile/<run>/report.ncu-rep \
   /home/meteorix/proj/<task>/profile/<run>/
```

Merging the REPORT.md without the binary loses the evidence behind it — the
reasoning survives, the ability to re-query the profile does not.

## 6. Escalate

When a task goes `stalled` or `blocked`, write into **Needs you** at the top of
`STATUS.md` — and make it a briefing, not an alarm:

- what was tried, with numbers
- what is now ruled out
- your read on where the runtime actually sits
- 2–3 concrete options with their costs

"04 is stuck" is a nag. Send a notification too
(`notify_on_needs_you`), once per escalation, not once per round.

## 7. Accumulate

Ask: did anything learned generalise beyond its task? If so add it to
`../kda/PLAYBOOK.md` as a numbered rule, citing the candidate that bought it —
rules 5, 8 and 9 all came from task 01 and now protect 02–04. Then propagate
across tasks explicitly: per-round workers have no memory of each other, so a
technique that worked on 01 reaches 02 only if you carry it in the brief.

A playbook rule that touches a `stalled` task's frontier returns it to
`optimizing`.

## 8. Close the cycle

Append one row to `<task>/cycles.jsonl`: spawn and report times,
outcome, hypotheses opened/closed, score, GPU seconds, `progress` true/false.
Rewrite that task's own `<task>/STATUS.md`, then refresh the deck roll-up line. Then **immediately respawn that task**
(§3) — do not batch it with the others.

Rewrite `STATUS.md`'s narrative sections and commit on the periodic sweep rather
than on every report; four tasks reporting independently means this handler runs
about four times as often as the old round close, so keep it cheap.

## 9. Halt conditions

Stop the loop, write a final summary, and say so plainly if:

1. all four tasks are `converged` / `stalled` / `blocked` — nothing left to
   schedule
2. `session.max_cycles` or `session.max_hours` reached — cost backstop
3. a guard tripped that implicates shared state (dirty bench submodule)

A single task stalling does **not** halt the loop; it stops getting workers and
the others carry on.

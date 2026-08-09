# Evolution trials — system design

**The method, not a trial.** This describes how a trial runs; what any particular
trial *found* lives in that trial's directory. It sits at the repo root beside
`../kda/PLAYBOOK.md` because both are accumulated method — a copy inside each trial
would either duplicate or diverge.

The pair divides cleanly: **`../kda/PLAYBOOK.md` is how to optimise a kernel**
(27 rules, each bought by a specific candidate); **this file is how to run many
agents doing that at once** (topology, the GPU lease, scheduling, the merge gate).

- `` — the frozen starting point a trial is materialised from
- `start-trial.sh` — materialises one
- `.claude/skills/evolve/SKILL.md` — the leader's procedure; this file is its reasoning
- `../trials/2026-08-08-t1/` — the first trial, and the source of most of
  the evidence below

**Implemented and run.** Written as a plan, then executed: trial 1 ran 34 cycles
across four tasks in a day. Where a section cites a number, that number was measured
during it. Sections marked *(superseded)* record a design that was tried and replaced,
because the replacement only makes sense next to what it replaced.

**Goal.** Start a leader session and the repo keeps optimising every task in the trial
on its own — one GPU, one worker per task, durable state on disk, human-readable
progress in each task's `STATUS.md`.

**Decided:** the human's session is the leader (§3). Tasks run continuously and
asynchronously (§8). The run is open-ended, halting on its own conditions (§8.5)
rather than waiting for a human between cycles.

---

## 1. What "self-evolving" has to mean here

Three properties, in priority order. Everything else in this document follows from
them.

1. **Resumable.** Agents are session-scoped; they die when the session ends. So a
   fresh leader session must be able to reconstruct the entire state of all four
   tasks *from disk alone*. No progress may exist only in an agent's context. This
   is the single hardest constraint and it drives §4, §5 and §8.
2. **Accumulating.** A loop that only produces kernels plateaus. The thing that has
   actually compounded in this repo is `../kda/PLAYBOOK.md` — rules 5, 8 and 9 were all
   *bought* with rejected candidates on task 01, and each one now prevents a class
   of wasted work on tasks 02–04. Promoting lessons into the playbook is a
   first-class output, not a nicety (§7).
3. **Honest.** An autonomous optimiser's characteristic failure is a number nobody
   reproduced, or a metric gamed rather than earned. The guards in §6 are the part
   of this design I would least want relaxed.
4. **Self-limiting.** A task that stops learning must notice, stop spending, and say
   so — with a diagnosis worth reading. Round budgets are dynamic and driven by
   measured progress, not a fixed count. §4.3.

---

## 2. The constraint that shapes everything

One RTX 2070 — the only serial resource, and the scoring instrument as well as the
compute device.

- **GPU wall-clock is the throughput ceiling.** A promotion-grade measurement on
  task 01 is 3–7 reps of a 6-shape sweep; tasks 03 (ctx 32768) and 04 (65536 envs)
  will be longer. The win comes from *maximising CPU-side work per GPU-second*, not
  from four-way parallel benching.
- **Parallelism corrupts the measurement.** `../kda/PLAYBOOK.md` rule 5 is the
  hardest-won lesson in this repo: the card drives the display, idles at 315 MHz,
  clocks cannot be locked under WSL2, and task 01's T=512 is *bimodal* across a
  1.64x clock-bin gap. Three agents running `nvcc` on the other 11 cores while a
  fourth benches is a new uncontrolled variable stacked on that. §6.4 is the answer
  and is the most important section here.

**Measured, and the first bullet is wrong.** Trial 2's trace (§6.6) puts GPU
occupancy at **~30% of wall clock** over its first 71 minutes — four agents
contending for one serialized lease still could not keep the card busy, and held time
is itself an upper bound because it includes compile and teardown. Queueing was not
steady contention either: a *single* 489-second lease accounted for nearly all of it,
against a 5-second median hold.

So the card is the only serial resource, but it is **not** the throughput ceiling.
The ceiling is the agent-side work — reading, deriving work models, writing kernels —
and the design consequence inverts: adding a second GPU would buy little, while
anything that shortens the think-and-write half buys directly. What survives intact is
the *second* bullet: serialisation is still required for measurement validity, which
is a correctness argument and does not depend on the card being saturated.

---

## 3. Topology

```
leader = this session   (main checkout /home/meteorix/proj, owns master)
  ├── worker-01  /home/meteorix/proj-wt/cb01   branch task/01-moe
  ├── worker-02  /home/meteorix/proj-wt/cb02   branch task/02-nsa
  ├── worker-03  /home/meteorix/proj-wt/cb03   branch task/03-decode
  └── worker-04  /home/meteorix/proj-wt/cb04   branch task/04-sps
```

Four `git worktree`s on ext4, created once. They share one `.git` object store, so
**worker branches are visible to the leader with no remote, no push, no fetch** —
the leader merges local branches directly.

Not using the Agent tool's `isolation: "worktree"`: it creates throwaway worktrees
and auto-removes them. These need stable paths and persistent named branches across
many rounds and many sessions.

Submodules are **not** initialised in the worktrees. All four workers reach the
bench through the single shared absolute path
`/home/meteorix/proj/kernelbench.com/benchmarks/cuda` — one venv, one set of problem
definitions, one `solution.py` slot per problem. The four problems are disjoint
directories, so those slots never collide.

### 3.1 Workers are per-cycle and independently paced

A worker is spawned to do *one* cycle — pick a hypothesis, build a candidate,
measure it, write the ledger, commit — then report and exit. It boots by reading
its task's state off disk.

Long-lived workers accumulate context that dies with the session, which directly
violates §1.1. Per-cycle workers make restart free: a fresh leader spawns a fresh
worker against the same on-disk state and loses nothing.

**No barrier between tasks.** Each task's next cycle starts as soon as *that
task's* worker reports — not when the slowest of four does. Round 1 measured the
cost of the barrier it replaces: three workers finished in 15, 16 and 17 minutes
and the fourth took 26, so every merge waited ~10 minutes on one task, and the
three finished tasks sat idle instead of starting their next cycle. There is no
cross-task dependency inside a cycle that a barrier would protect — the task
directories are disjoint by construction (§9) and the GPU is already serialised
per-command by the lease (§6), which is a much finer grain than a round.

This is the same pipeline-over-barrier argument that applies to any fan-out:
wall-clock becomes the slowest single *chain*, not the sum of the slowest member
of each stage.

**Cycle length is a dial, not always 1.** A worker may be given a small cycle
budget (`worker.cycles_per_spawn`, default 1) and loop internally that many times,
committing after each. Two cycles amortise the ~1–2 minutes of boot reading over
more work; beyond about three the context growth starts to defeat the point, and
a crash loses more. Raise it only for tasks whose cycles are short and mechanical.

### 3.2 Path fix required first — done

*Applied during trial 1; kept for the reasoning.* `run.sh` hardcoded
`WORK=/home/meteorix/proj/workspace/cuda-bench`. From a worktree
that resolves back to the *main* checkout, so a worker's new candidate would be
invisible to `run.sh use` — every worker would silently bench the leader's copy of
the code. Fix it to the pattern `profile.sh` and `repeat.sh` already use:

```bash
WORK=$(cd "$(dirname "$0")" && pwd)
```

`BENCH` stays absolute and shared. This is the highest-priority prerequisite in the
whole plan.

---

## 4. Durable state — the resume contract

A fresh leader session reconstructs everything from these files. Nothing else is
authoritative.

| File | Owner | Role |
| --- | --- | --- |
| `STATUS.md` | leader | **The file you read.** Dashboard: champions, deltas, what each task is doing, what needs you. Overwritten each round. §5 |
| `<task>/HYPOTHESES.md` | worker | The task's open idea backlog and its *closed* frontier, with reasons. The resume point for "what to try next". §4.1 |
| `<task>/candidates.jsonl` | worker | Append-only ledger — already the repo's best artefact. Every candidate, promoted or rejected, with measurements and a `note` saying why |
| `<task>/docs/report.md` | worker | The narrative for a human reader |
| `<task>/cycles.jsonl` | task | Append-only per-**cycle** record, keyed by (task, cycle): spawn/report times, outcome, hypotheses opened/closed, geomean, GPU seconds, progress. The data the stall detector runs on. §4.3, §8.4. (`rounds.jsonl` is its synchronous predecessor, kept as history) |
| `evolve.config.json` | you | Stall thresholds and scheduling dials, tunable without touching the skill. §4.3 |
| `../kda/PLAYBOOK.md` | leader | Accumulated cross-task lessons. §7 |
| `<task>/CONTRACT.md` | task | That task's contract **and its hand-counted work model** (§4.4). Self-contained: the machine/bench preamble is duplicated into each so a worker never reads a sibling |
| git history | leader | The archive. `STATUS.md` keeps only the recent window |

**Invariant:** a worker has made no progress until it is written to disk. A worker
that finds something and exits without committing or updating `HYPOTHESES.md` has
burned GPU time for nothing. This goes in every worker brief.

### 4.0 Everything a task needs lives in its own directory

**Changed after the async switch.** Centralised `CONTRACTS.md`, `STATUS.md`,
`cycles.jsonl` and `evolve.config.json` were a hangover from synchronous rounds:
one file per concern, four tasks inside it. With independent cycles that shape is
wrong — two tasks writing one status file is a merge conflict waiting to happen,
and a worker had to read three other tasks' contracts to find its own.

Now each task directory holds `CONTRACT.md`, `STATUS.md`, `HYPOTHESES.md`,
`cycles.jsonl`, `config.json`, `candidates.jsonl`, `docs/`, `candidates/`,
`runs/`, `profile/`, plus whatever task-local tools it writes. **A worker needs its
own directory, the shared wrappers, and the playbook — nothing else.**

Three things stay shared, and the reason is not tidiness:

- **`gpu.sh`.** There is one GPU. The lease cannot be per-task.
- **`run.sh` / `ab.sh` / `repeat.sh` / `profile.sh`.** In a single day, six harness
  bugs were found by one task and fixed for all four: a promotion bias in the A/B
  (arm A's first rep ~32% slow on a cold lease), a hardcoded `moe_*` regex in the
  profiler, two clobbering bugs, an exit-code bug that masked the wrapped command's
  status, and an env-scrubbing self-deadlock. Per-task copies would have fragmented
  every one. Tasks add local tools freely and already do — `sched.py`, `floors.py`,
  `determinism.py`, `step_profile.sh` each live in the task that wrote them.
- **`../kda/PLAYBOOK.md`.** Cross-task transfer is the compounding asset. Two tasks
  independently reached the same conflict-cost rule from opposite directions on the
  same afternoon.

The deck `STATUS.md` becomes a roll-up with no authority: if it disagrees with a
task's own file, the task's file wins.

### 4.0b Re-entering from scratch — the trial is the boundary

An earlier design reset a single task in place, archiving into `<task>/attempts/NN/`.
That is **superseded**: the unit of re-entry is a whole trial, materialised from the
frozen starting point by `../cuda-bench/start-trial.sh`. One mechanism, not two.

The reasoning survives the change. "From scratch" is only meaningful against a
baseline — a run reaching 75% of a DRAM floor means one thing if the previous run
reached 30% and another if it reached 76% — so the starting line has to be fixed and
the previous run has to be kept. `../cuda-bench/README.md` records exactly what
carries between trials and what does not, and the one genuine judgement call: the
playbook carries by design, so two trials are **not** a controlled comparison of the
loop alone unless a revision is pinned in `TRIAL.md`.

The starting contracts deliberately omit the hand-counted work model (§4.4). Rule 2's
evidence is that deriving it early is what corrects a task's hypotheses before any GPU
time; shipping it pre-computed would hand every later trial that correction, and
nobody would learn whether the loop produces it.

### 4.1 `HYPOTHESES.md` — the backlog that makes the loop non-random

Without this, an autonomous loop reinvents ideas it already killed. Per task, two
lists:

**Open** — each entry: the hypothesis, where it came from (profile, SASS, playbook,
contract arithmetic), predicted effect and on which shapes, estimated cost, and what
would falsify it.

**Closed** — each entry: the hypothesis and *why it is closed*. Closing is as
valuable as opening. Task 01 already has real examples that must be seeded here:

- `T=1 is closed` — the `c6` note: 278 us of kernel against 113 MB / 414 GB/s =
  273 us, within 2% of the measured DRAM roof. "No further candidate should be spent
  on it." An autonomous loop that doesn't know this will spend several.
- `gemm1 occupancy via __launch_bounds__ is closed` — `c9`, rejected at −15%; the
  64 registers of accumulator fragments are the floor.
- `shared-store bank conflicts via thread remap is closed` — `c7`, rejected at
  −16%; and the stall profile showed short_scoreboard is 2.5% of the stall budget,
  so the conflict was real and nearly free.
- `Open, named, large`: `mma.sync` with hand-addressed fragments allowing XOR
  swizzle; or BK=64 with BN=32. Both are rewrites, not tweaks.

Seeding these four tasks' backlogs from the existing ledger is part of phase 0.

### 4.2 Task lifecycle

`ramping` → `optimizing` → `frontier-thin` → `converged`
                  ↕
              `stalled` (escalated to you) · `blocked` (waiting on you)

- **ramping** — no validated harness yet. Playbook rule 10 order: harness →
  reference validated against a slow obviously-correct ground truth → broken-kernel
  self-test asserted FAIL → first candidate. Tasks 02/03/04 start here.
- **optimizing** — open hypotheses with predicted effect above the noise floor.
- **frontier-thin** — only speculative or very expensive hypotheses remain. Reduce
  to one worker slot per N rounds; spend the freed GPU elsewhere.
- **converged** — no open hypothesis worth its GPU cost. Stop scheduling it. This is
  success, not a stall, and must never be reported as one.
- **stalled** — the budget in §4.3 ran out with no frontier movement. Stop
  scheduling, write a diagnosis under **Needs you**.
- **blocked** — a worker hit something only you can settle. Immediate, no waiting.

When all four reach `converged`, `stalled` or `blocked`, the leader writes a final
summary and **stops**. A self-evolving system that cannot decide it is done just
burns tokens and GPU.

### 4.3 Progress, stall detection and the dynamic budget

**Progress is frontier movement, not champion movement.** This distinction is the
whole design. A round counts as progress for a task if *any* of:

- the champion improved (strongest signal), **or**
- a hypothesis was **closed with evidence** — a candidate ran, measured, and ruled
  something out, **or**
- new hypotheses were **opened from evidence** — a profile, a SASS read, or an
  arithmetic floor produced a concrete next thing to test, **or**
- the task advanced a `ramping` milestone (harness → reference validated →
  self-test FAILing).

Judging by promotions alone would have flagged task 01 as stuck across `c7` and
`c9` — both rejected, both large regressions, and both among the most valuable
rounds in the ledger. `c7` produced the stall-attribution measurement that
"redirected the whole effort", and `c9` closed occupancy permanently in seconds of
`-Xptxas -v`. **A rejected candidate that closes a hypothesis is a good round.**

**The counters** live in `rounds.jsonl` and reset on any progress:

| Counter | Default trigger | Rationale |
| --- | --- | --- |
| **cycles since progress** | 6 | per task, not per round — cycles are now independent |
| wall-clock since progress | 4 h | what you asked for |
| ~~GPU-minutes since progress~~ | ~~90~~ → **12** | **recalibrated after round 1.** 90 was set on the theory that GPU is the scarce resource; measured, four workers held the lease ~6 minutes total across ~64 worker-minutes, under 10% utilisation. At ~1.5 GPU-min per cycle a task needed ~60 cycles to trip it — the counter was effectively dead. 12 makes it fire at ~8 barren cycles, in the same range as the others |
| consecutive build/`check.py` failures | 3 | stuck on mechanics, not ideas — detect early and cheaply |
| **single tool call / lease hold** | 5 min | new. A worker stuck inside one call is invisible to a cycle-boundary detector; round 1 lost ten minutes to exactly this |

Whichever fires first. All four live in `evolve.config.json`, per task, so a
long-cycle task like 03 (ctx 32768) can carry a larger budget than 01.

**Stall classes get different responses** — a single "stuck" verdict would be
useless:

| Class | Signature | Response |
| --- | --- | --- |
| **Barren** | candidates run and measure cleanly, nothing opens or closes | soft → hard escalation |
| **Broken** | 3 consecutive candidates fail to build or fail `check.py` | escalate early; this is mechanics, and a human unsticks it in a minute |
| **Thrash** | same hypothesis re-tested under new candidate ids; champion oscillating | escalate; the loop has lost the plot |
| **Exhausted** | backlog empty, or every open hypothesis predicts below the noise floor | **not a stall** — mark `converged`, report as success |
| **Blocked** | ambiguity only you can settle (see §12.2) | immediate escalation, no budget consumed |

**Escalation is graded.** At **50% of budget** the leader intervenes itself before
bothering you, because the most common stall is a worker that stopped measuring and
started guessing — playbook rule 6. In order:

1. **Force a profile.** `ncu` is the highest-value information available and three
   of task 01's promotions trace directly to one. Do not let a task stall without a
   profile of its current champion.
2. **Re-read the contract and the arithmetic floor** — `tiles.py`-style. Task 01's
   T=1 frontier was closed by arithmetic, not by a candidate.
3. **Re-frame with a fresh worker** and an explicitly different angle, since
   per-round workers carry no memory of the previous framing.

At **100%** the task goes `stalled`, scheduling stops, the GPU goes to live tasks,
and the leader writes a diagnosis under **Needs you** in `STATUS.md` containing:
what was tried and measured, what is now ruled out, the leader's read on the
blocker, and 2–3 concrete options with their costs. An alarm that just says "02 is
stuck" is a nag; this should be a briefing you can act on in a minute.

**Resume.** `stalled` and `blocked` are not terminal. A task returns to `optimizing`
when you answer, or automatically when a playbook rule promoted from another task
(§7) touches its frontier — cross-task transfer is exactly the kind of thing that
unsticks a barren task without your involvement.

---

## 5. `STATUS.md` — the progress file

Leader-owned, rewritten at the end of every round, designed to be read cold.

```markdown
# cuda-bench — status
Round 47 · updated 2026-08-08T14:32Z · leader session active

## Needs you  (2)

**04 grid-mingru-sps — STALLED** (barren: 6 rounds, 3h50m, 94 GPU-min, no frontier move)
- Tried: c8 fused env_step+policy (−3%), c9 warp-per-env (−11%), c10 same at 2 envs/warp (−9%)
- Ruled out: fusion across the env boundary — the LCG respawn serialises it, all three
  variants pay the same divergence. Closed in HYPOTHESES.md.
- My read: SPS is bounded by the h=256 MinGRU recurrence, not by env cost. breakdown
  puts 78% of the step in the 3 GRU layers. The remaining backlog is all env-side, so
  it is aimed at 22% of the runtime.
- Options: (a) persistent-block policy kernel keeping h resident across the horizon —
  large rewrite, ~4 rounds, the only thing that touches the 78%; (b) accept 0.412 and
  mark converged; (c) fp16 the recurrence — but positions/rewards must stay bit-exact,
  so this needs your read on whether logits atol 1e-3 survives it.

**03 megaqwen-decode — BLOCKED** (round 45, no budget consumed)
- `decode_steps` semantics ambiguous vs PROMPT.txt: is the KV cache advanced before or
  after the last step's logits? Both readings pass check.py at ctx 128. Correctness, so
  parked rather than assumed.

## Tasks
| Task | State | Champion | Geomean | Δ round | Δ total | Cands | Open hyp | Since progress |
|------|-------|----------|---------|---------|---------|-------|----------|----------------|
| 01 glm52-fused-moe | optimizing    | c11 | 0.1189 | +2.1% | +8.6% | 12 | 3 | 0 rounds |
| 02 deepseek-nsa    | optimizing    | c2  | 1.84x  | 0%    | —     | 3  | 5 | 2 rounds / 41 GPU-min |
| 03 megaqwen-decode | blocked       | —   | —      | —     | —     | 0  | — | — |
| 04 grid-mingru-sps | stalled       | c7  | 0.412  | 0%    | +14%  | 10 | 1 | 6 rounds / 3h50m |

## This round
- 01 promoted c11 (mma.sync + XOR swizzle): 0.1142 → 0.1189, ms_min wins 5/6 shapes
- 02 rejected c3 (block-mean fused into scoring): −4%; hypothesis closed, counter reset
- 04 stall budget exhausted → escalated (soft escalation at round 44 forced an ncu
  profile; it produced the 78% figure above but no new candidate)

## GPU
Lease utilisation 71% · longest hold 14m (01 ncu --set full) · no starvation
04's freed slot reallocated to 01

## Recent rounds
… last 10, one line each …
```

Note what the escalation is *not*: "04 is stuck". It is a briefing — what was
measured, what is closed, where the runtime actually sits, and options with costs.
That is the output the stall detector exists to produce.

Full history stays in git and `candidates.jsonl`; `STATUS.md` keeps a window so it
stays readable.

---

## 6. GPU serialisation

### 6.1 The lease

New `harness/gpu.sh`; every GPU-touching command goes through it, no
exceptions:

```bash
gpu.sh <label> -- <command...>
```

- `exec 9>/tmp/cuda-bench.gpu.lock; flock -x 9` — held for the child's whole
  lifetime, released by the kernel on exit *or crash*, so a killed worker (or a
  killed session) cannot leak the lease.
- Write `<pid> <label> <task> <iso8601>` to `/tmp/cuda-bench.gpu.holder` while held;
  append acquire/release/duration to `/tmp/cuda-bench.gpu.log` — that log is what
  feeds `STATUS.md`'s GPU line.
- `-w <seconds>` wait budget; exit 75 on timeout so callers back off into CPU work
  rather than blocking.

`run.sh check|bench`, `profile.sh`, `repeat.sh` and the new `ab.sh` all route
through it. Nothing else touches the GPU.

### 6.2 Two preflight guards, inside the lease

Run after acquiring, abort the run on failure:

1. **No foreign GPU processes** — `nvidia-smi --query-compute-apps=pid,used_memory`
   clear of cuda-bench processes. 8 GB VRAM with a leftover context is an OOM, not a
   slow run.
2. **Bench submodule integrity** — `git -C kernelbench.com status --short` shows
   nothing outside `solution.py` / `__pycache__`. Safety-critical: `check.py`,
   `benchmark.py`, `shapes.py`, `reference.py`, `problem.yaml` and `PROMPT.txt` are
   bench-rule-forbidden to edit, they live in a submodule *shared by all four
   workers*, and an edit by worker 03 silently invalidates every number workers
   01/02/04 produce. Checking at lease time catches it in minutes instead of at
   merge time.

### 6.3 Keep work off the lease

Hard rule: **a candidate does not enter the GPU queue without a written hypothesis
and a `-Xptxas -v` result.** Playbook rule 6 as scheduling policy — better science
*and* less contention.

Lease-free: writing the candidate; `nvcc -Xptxas -v -arch=sm_75 -cubin` for
registers/shared/spills (**no GPU needed** — it killed `c9` in seconds);
`cuobjdump -sass` / `nvdisasm` (the `c8` win came from reading SASS);
`ncu --import <existing>.ncu-rep --page details|source|raw` (re-reading a captured
report needs no device); ledger, `HYPOTHESES.md`, docs, rebases, merges.

Lease-required: `check.py`, `benchmark.py`, `repeat.sh`, `ab.sh`, `profile.sh`, any
`smoke.py` that launches a kernel.

**Optional:** `run.sh prebuild <nn>` — import the solution module with
`CUDA_VISIBLE_DEVICES=` and `TORCH_CUDA_ARCH_LIST=7.5` to populate
`/root/.cache/torch_extensions` without creating a CUDA context, so the leased run
starts already built. Worth it (nvcc is a large fraction of a short bench) but
**needs verifying** that the import path never touches `torch.cuda`. Falls back
cleanly to compiling under the lease.

Cap `MAX_JOBS=2` per worker for ninja, so four concurrent builds fit 12 cores with
headroom for whoever holds the lease.

### 6.4 Making the measurement survive the parallelism

**a. Paired A/B inside one lease hold.** New `ab.sh <nn> <champ> <chal> <reps>`
interleaving `champ, chal, champ, chal, …` within a single lease. Today `repeat.sh`
runs a candidate's reps in one block, so a champion measured an hour ago is compared
against a challenger measured under different clocks and different ambient CPU load.
Interleaving makes both arms sample the same conditions and turns promotion into a
paired comparison. Given the T=512 bimodality this is strictly better than what
exists now, parallelism aside.

**b. Rank on `ms_min` alongside the median.** Already this repo's own conclusion
(`measurement-2`: "a median over reps is the wrong statistic for a bimodal
distribution — it reports which bin was drawn more often"). Under four-agent load
the fast mode is the cleaner signal.

**c. Record ambient load in the ledger.** Every `candidates.jsonl` entry gains
`"ambient": {"agents_active": N, "loadavg": X}` at lease time. Absolute geomeans
stop being comparable across sessions the moment agents run in parallel; recording
this keeps the ledger honest instead of quietly wrong.

**d. One quiet run for anything headline.** Before a number enters `docs/report.md`,
`STATUS.md` as a champion, or the README, the leader re-measures with all other
workers idle. It is the only way the deck's numbers stay comparable to the c0–c8
history.

### 6.5 Fairness

`flock` is not FIFO, so four contenders can starve one — and `ncu --set full`
replays ~34 passes, a long hold. `gpu.sh` logs every acquire with its duration and
the leader reviews it each round; the result lands in `STATUS.md`. Escalate to a
ticket queue only if starvation actually shows up.

Measured in t2: starvation did **not** show up. Total queue time was real but
concentrated in one long hold rather than spread across contenders, and no worker was
repeatedly passed over. The ticket queue stays unbuilt.

### 6.6 Reading the lease log back — the trial trace

`harness/timeline.py` renders a whole trial as one page: a lane per task on a shared
time axis, GPU holds and queue blocks, commit ticks, merge rules, cycle bands. It
joins three logs the system already writes — the lease audit log, per-branch
`git log`, and the merge commits on master — and adds no new instrumentation.

```bash
./timeline.py                 # writes ./timeline.html next to the trial
```

It exists because the other artefacts each answer a different question and none
answers this one. `wstat.sh` answers *is a worker alive right now*. A task's
`STATUS.md` answers *what did this worker conclude*. Neither shows **where the time
actually went**, and that turns out to be where the design-level surprises are: §2's
occupancy correction and §6.5's starvation verdict are both readings of this trace,
and neither was visible in any per-cycle report.

Run it at least once per trial, and treat occupancy as an upper bound — a lease covers
process start, compile and teardown, not just kernel execution.

---

## 7. Accumulation — the part that compounds

At the end of each round the leader asks: *did anything learned generalise beyond
its task?* If yes it goes into `../kda/PLAYBOOK.md` as a numbered rule, with the
candidate that bought it cited. Rules 5, 8 and 9 all came from task 01 this way and
now protect 02–04.

The leader also propagates *across* tasks within a round — a stall-attribution
technique that worked on 01's gemm1 is worth handing to 02 explicitly, because
per-round workers have no memory of each other.

This is the difference between a system that gets better and a system that just runs.

---

## 8. The leader loop — event-driven, not round-based

There is no round. The leader is a small handler that runs **once per worker
report**, plus a periodic sweep. Everything is per-task.

### 8.1 On a worker report (the hot path)

Triggered by that task's worker completing. Touches **one** task:

1. **Merge gate** (§9) for that task only — including re-running `check.py` and a
   bench rep yourself on any promotion.
2. **Merge** its branch to master, `--no-ff`. Archive any new `.ncu-rep` into the
   leader checkout (§9).
3. **Append `<task>/cycles.jsonl`** — one row per worker cycle: cycle number, spawn
   and report times, outcome, hypotheses opened/closed, GPU seconds, progress
   true/false.
4. **Update that task's row** in `STATUS.md`.
5. **Decide whether to respawn that task** (§8.3) and do it immediately. Do not
   wait for the other three.

This is deliberately small. It must be cheap enough to run four times as often as
the old round close, because it now will.

### 8.2 The sweep (the cold path)

On a timer, and always before spawning:

- Run the stall detector (§4.3) over each `<task>/cycles.jsonl`, for **every** task including
  ones with a worker in flight — a task whose worker is mid-cycle can still be
  past its wall-clock budget.
- Liveness check: process table by worktree, lease holder and hold duration. Flag
  any single lease hold or single tool call past a threshold. **Round 1 needed
  this and did not have it** — a worker sat ten minutes inside one `find /` call,
  invisible to a detector that only looks at cycle boundaries, and the leader
  twice misread liveness from file mtimes, which say nothing once a worker stops
  writing and starts running.
- Accumulate (§7): promote lessons to `../kda/PLAYBOOK.md`, propagate across tasks.
  This is the one genuinely cross-task step, and it is why the sweep exists at all
  rather than folding into §8.1.
- Rewrite the `STATUS.md` narrative sections and commit.

### 8.3 Respawn policy

After a task's worker reports, spawn its next cycle unless:

- the task is `converged`, `stalled` or `blocked`;
- a global halt condition fired (§8.5);
- **backoff applies.** A cycle that ends in under `worker.min_cycle_seconds`
  (default 180) *without* progress is treated as a fast failure: wait, and after
  `worker.fast_fail_backoff` consecutive ones, stop scheduling the task and
  escalate. Without this, an agent that errors out in twenty seconds respawns
  forever and burns tokens at the speed of the failure. The old round barrier
  masked this risk by pacing everything to the slowest worker; removing it exposes
  it, so the guard has to go in at the same time.

**Concurrency:** at most one worker per task, so at most four. That is a policy,
not a limit — round 1 held the GPU under 10% of wall clock, so the machine could
carry more. Two workers on one task's independent hypotheses is now possible and
is left as §12.7 rather than assumed.

### 8.4 What this changes about `rounds.jsonl`

It becomes `<task>/cycles.jsonl`, one file per task rather than one keyed by round.
Anything that is not a cycle — the phase-0 build, a regression check, a change to the
scheduler itself — belongs in a git commit message, not in a ledger. The stall
detector always operated per task, so it reads *better* against per-task cycles;
the round column was an artificial unit that only existed because the scheduler
was synchronous. Existing `rounds.jsonl` rows are kept as history.

### 8.5 Halt conditions

Checked in the sweep and before each respawn:

1. Every task is `converged` / `stalled` / `blocked`.
2. Session cap reached (`session.max_hours`, and `session.max_cycles` replacing
   `max_rounds`).
3. A guard tripped that implicates shared state — a dirty bench submodule stops
   everything, because it invalidates every task's numbers at once.

### 8.1 Boot

The harness is a **skill**: `.claude/skills/evolve/SKILL.md`, invoked as `/evolve`.
It encodes steps 1–9, the guards, the stall detector and the worker brief template.
Starting the system is then one command in a fresh session, and the resume contract
in §4 is what makes that work.

**Unattended is the decided mode.** `/loop /evolve` drives rounds back-to-back,
self-paced — a round ends when its workers report, so a fixed interval would either
truncate rounds or idle between them.

Note this still requires the session to stay open; agents are session-scoped.
"Unattended" means you can walk away, not that it survives closing the terminal. If
the session does die, §3.1 and §4 mean the loss is bounded to the round in flight —
a fresh `/evolve` resumes from disk.

**Halt conditions**, checked at the top of every round:

1. All four tasks are `converged` / `stalled` / `blocked` — nothing left to schedule.
   Write the final summary and stop.
2. Session round cap or wall-clock cap reached (`evolve.config.json`, proposed
   default 24 rounds or 12 h, whichever first). A pure cost backstop — the loop
   reports where it got to and stops rather than running until you notice.
3. A guard tripped in a way that implicates shared state — a dirty bench submodule
   (§6.2) is a stop-everything condition, not a per-task one, because it invalidates
   every task's numbers at once.

A single task stalling does **not** halt the loop. It stops getting workers, its GPU
slot goes to a live task, and the loop continues.

**Notification.** When a task lands in **Needs you**, push a notification rather than
leaving the briefing sitting in `STATUS.md` — the point of "stop and let me know" is
that it reaches you. One notification per new escalation, not per round.

---

## 9. Git protocol and the merge gate

**Ownership.** A worker commits *only* under its own `<NN>-*/`.
Everything else — `run.sh`, `gpu.sh`, `ab.sh`, `profile.sh`, `repeat.sh`,
`../kda/PLAYBOOK.md`, `README.md`, the deck `STATUS.md` roll-up, the shared harness
template — is **leader-only**. A worker that wants a shared file changed asks; it
does not edit. This makes the conflict surface between branches ~zero by
construction.

**Commits**, matching the existing history style, always carrying the numbers:

```
cuda-bench 02: harness + reference validated vs CPU float64 ground truth
cuda-bench 01: promote c11, 0.1142 -> 0.1189 (4-rep paired A/B, ms_min 5/6)
cuda-bench 04: reject c7, -6%; hypothesis closed
```

Rejected candidates are committed too — the ledger's value is that `c3`, `c4`, `c7`
and `c9` are all recorded with *why*.

**Rebase.** Workers `git rebase master` before every commit and when told master
moved. **Merge.** Leader merges `--no-ff`, one branch at a time, then broadcasts.

**The merge gate — the leader is not a rubber stamp.** Before merging a promotion:

- `candidates.jsonl` entry with `check: PASS`, `framework` ≠ `pytorch_only`,
  `reps >= 3`, `ms_min` present, `ambient` recorded, and a `note` explaining *why*
  it won
- `git diff --stat master..task/NN-*` touches nothing outside the task dir
- `git -C kernelbench.com status --short` clean
- **the leader re-runs `check.py` and at least one bench rep itself.** A number only
  the worker that produced it has ever seen is the one failure that costs the whole
  run its credibility. Budget lease time for it.
- the candidate names which hypothesis it tested, and that hypothesis gets opened or
  closed in `HYPOTHESES.md` either way

---

## 10. Phase 0 — leader, blocking, before any worker spawns

1. `run.sh`: make `WORK` script-relative (§3.2). Verify from a scratch worktree that
   `run.sh use` installs *that worktree's* candidate.
2. Write `gpu.sh` — lease, holder, log, both preflight guards.
3. Route `run.sh check|bench`, `profile.sh`, `repeat.sh` through it.
4. Write `ab.sh` — paired interleaved A/B, min/median/max per shape per arm.
5. Optional: `run.sh prebuild`, after verifying the no-CUDA-context claim.
6. **Extract a shared harness template from task 01** — `collect.py`,
   `prof_driver.py`, `smoke.py`, the `candidates.jsonl` schema, the
   `docs/draft.md`/`report.md` skeleton. Three workers are about to build the same
   scaffolding; if they invent three ledger formats the deck stops being one deck.
   Highest leverage item after the path fix.
7. Seed the four `HYPOTHESES.md`, including task 01's closed frontier (§4.1) — this
   is what stops the loop re-spending candidates on T=1.
8. Write `evolve.config.json` with the §4.3 thresholds, and the first `STATUS.md`
   and `rounds.jsonl`.
9. Write `.claude/skills/evolve/SKILL.md` (§8.1), including the stall detector and
   the mandated escalation format.
10. Create the four worktrees and branches under `/home/meteorix/proj-wt/`.
11. **Regression-check task 01**: re-run `c8` through the new `gpu.sh`/`ab.sh` path
    and confirm it still reproduces 0.1142. If the plumbing moved the numbers, find
    out now, not at merge time.

---

## 11. Failure modes

| Risk | Mitigation |
| --- | --- |
| Session ends mid-round, progress lost | Per-round workers (§3.1) + write-to-disk invariant (§4) |
| Loop re-spends candidates on a closed frontier | `HYPOTHESES.md` closed list (§4.1) |
| Loop runs forever with nothing left to win | `converged` state; leader stops (§4.2) |
| Loop grinds for hours learning nothing | Dynamic budget on rounds / wall-clock / GPU-minutes, graded escalation (§4.3) |
| **Stall detector fires on a task that is doing fine** | Progress = frontier movement, not promotions — `c7` and `c9` were rejected *and* valuable (§4.3) |
| **`converged` misreported as `stalled`** | Exhausted backlog is a distinct class and reports as success (§4.3) |
| Escalation arrives as an unactionable nag | Diagnosis format is mandated: tried / ruled out / read / options with costs (§5) |
| Worker edits a bench-forbidden file in the shared submodule | Guard at every lease acquisition (§6.2) + merge gate |
| Numbers shifted by ambient load from other workers | Paired A/B (§6.4a), `ms_min` (b), ambient recorded (c), quiet re-run for headlines (d) |
| Workers diverge from deck conventions | Shared harness template (§10.6) |
| Worker benches the main checkout's code | `run.sh` path fix (§3.2), verified in phase 0 |
| VRAM OOM from a leaked context | `nvidia-smi` preflight (§6.2) |
| Long `ncu` hold starves others | Lease log reviewed each round, surfaced in `STATUS.md` (§6.5) |
| A number nobody reproduced | Leader re-runs `check.py` + a bench rep before merging (§9) |
| Windows-side GPU activity (display, other apps) | Unfixable, pre-existing; paired A/B is the defence |

---

## 12. Open decisions

1. **Stall thresholds.** Defaults proposed in §4.3 are 6 rounds / 4 h / 90 GPU-min /
   3 consecutive mechanical failures, per task, whichever fires first. The 4 h is
   yours; the other three are my guesses and are the ones to argue with. In
   particular, **GPU-minutes is the dial I would actually tune** — it is the only
   scarce resource, and unlike wall-clock it does not punish a task for being
   descheduled while another holds the lease.
2. ~~**Unattended or round-at-a-time?**~~ **Decided: unattended** (§8.1). Residual
   dial: the cost backstop in halt condition 2 — proposed 24 rounds or 12 h per
   session, whichever first. A guess; adjust in `evolve.config.json` once a round's
   real duration is known, which won't be clear until a few have run.
3. **`ncu` budget.** Profiling is the longest lease hold and the highest-value
   information — three of task 01's promotions trace to a profile, and it is step 1
   of soft escalation. Reserve a slot per task per N rounds, or first-come?
4. **How autonomous on ambiguity?** When a worker hits something like an unclear
   reference semantic, does it park the task as `blocked` or pick the most
   defensible reading, record the assumption, and carry on? *Recommend: park
   anything affecting correctness, proceed on anything affecting only performance.*
5. **Task 01's next candidate is a rewrite**, not a tweak — `mma.sync` with
   hand-addressed fragments, or BK=64/BN=32. That is several rounds with no promotion
   and a real chance of ending at −10%. Is the system allowed to spend that, or
   should it prefer cheap candidates until 02–04 catch up? *Recommend: allow it, but
   only once 02–04 are out of `ramping`, so a long shot isn't holding the only live
   frontier.*
6. **Worktrees vs one shared checkout** — worktrees proposed and assumed; say if
   you'd rather not have four extra checkouts on disk.

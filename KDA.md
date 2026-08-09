# KDA Playbook

A supplement to [Kernel Design Agents](kernel-design-agents/), not a summary of it.
Read the source first — it is short:

- [README.md](kernel-design-agents/README.md) — setup, minimal flow, workspace layout
- [docs/agent-flow.md](kernel-design-agents/docs/agent-flow.md) — the loop, task
  contract, evidence records, promotion rule
- [prompts/basic-flow.md](kernel-design-agents/prompts/basic-flow.md) — the starter
  prompt and its fill-in contract

This file records only what those documents do not say: what running the loop on this
machine actually cost, and the rules that came out of getting things wrong.

**Paths here are relative to the repository root** — the rig, or a trial's top directory.
Inside a trial this file is symlinked into every task directory, so from a task dir prefix
them with `../`.

Worked examples:

- [workspace/fp16-gemm/docs/report.md](workspace/fp16-gemm/docs/report.md) — fp16 GEMM
  taken from a naive kernel to 88% of cuBLAS on sm_75. Private harness.
- [trials/2026-08-08-t1/01-glm52-fused-moe/docs/report.md](trials/2026-08-08-t1/01-glm52-fused-moe/docs/report.md)
  — GLM-5.2 fused MoE against the [KernelBench deck](KERNELBENCH.md), where the
  validator, shapes and score belong to the bench rather than to the loop.

**When it pays:** the loop's value is that it refuses to skip steps, not that any step
is novel. For a one-off kernel you already understand, the ceremony is overhead. It
pays when the search space is wide enough that you will forget what you already tried.

---

## Rules from the first run

1. **Fill in the contract before starting the agent.** KDA puts this at step 2, before
   the session — but nothing enforces it. In the demo the agent invented the shapes,
   the tolerance, *and* the 60% performance target, then reported success against its
   own target. Require the agent to state any field it invents and stop for approval.
2. **Add two fields the contract template lacks.** First: the real shapes or workloads
   that matter — round numbers are the agent's default and may not be your workload.
   Second, and added after four tasks: **a hand-counted work model, computed before any
   kernel exists.** Executed FLOPs and executed bytes per shape, the arithmetic
   intensity, which roof therefore binds, and the resulting floor in ms.
   **Do not take the numerator from `problem.yaml`.** Its formulas are a *scoring*
   convention — fixed so submissions stay comparable — not a model of what your kernel
   does, and on this deck they were wrong in every way available: the MoE bytes formula
   charges all 129 experts at every T and **over-counts traffic 14.3× at T=1**; the NSA
   flops formula is dense S² a correct sparse kernel never executes; the decode and grid
   problems supply no work model at all, only a stated constant.
   The evidence that this belongs at contract time: the two tasks that computed floors
   in cycle 1 had their main hypotheses **corrected before any GPU time** — one closed
   tensor cores outright, the other discovered it was compute-bound by 10× and that its
   whole backlog was aimed at the wrong resource. The two tasks that did not spent
   candidates first and found the number at cycle 5.
   Two properties the field needs. The roof must be **measured on this machine for this
   access pattern** (414 GB/s streaming, 435.3 for one task's pattern, never the 448
   datasheet). And the table must be **recomputed once c0 exists**, because the
   algorithm's floor and the kernel's floor are different numbers — see rule 22 for what
   happens when a contract-time model is trusted after it stops describing the code.
3. **Validate the reference before any candidate.** Check the library reference against
   a slow, obviously-correct ground truth (CPU `double`, small shape) first. A layout or
   transpose mix-up otherwise makes every later number wrong while everything "passes".
4. **Ship a deliberately broken kernel in the harness** and assert it FAILS. A validator
   that never fails is worse than no validator.
5. **Warm the clocks before measuring**, and report medians. On an idle consumer GPU the
   first-measured kernel eats the ramp — this briefly made the demo's kernel look like
   it beat cuBLAS. **Then repeat the whole run and compare medians across runs**, and
   do it before spending candidates on a gap. Warming and medianing *within* a run is
   not enough: on the MoE task it left T=1 drifting 37% run to run, which was wider
   than the gap between the two candidates it was being used to rank, and a single run
   promoted the wrong one. Clock locking would be the real fix and is not available
   here — `nvidia-smi -lgc` returns `Unknown Error` under WSL2 against the Windows
   driver. Watch for this hardest where a score is a geomean over shapes of very
   different size: the smallest shape is usually both the noisiest and the one the
   geomean is most sensitive to. **And check whether a shape is bimodal rather than
   noisy — reps do not fix that.** One MoE shape had two modes 1.64x apart, matching
   the card's boost range; over seven reps its median reported *which clock bin came up
   more often*, and flipped the ranking of two candidates whose fast modes were 1.3%
   apart. Keep min and max next to the median in the CSV: reading `ms_min` is what
   separated them.
6. **Measure before theorizing.** `nvcc -Xptxas -v` gives registers and shared memory in
   seconds and settled a regression immediately. **Read its stack-frame column, not just
   its spill columns:** `64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
   is *not* harmless. A runtime-indexed local array is not a spill, is not counted as
   one, and costs the same — on the NSA task it put **97.30% of all L1TEX sectors in
   local memory** at 0.4 of 32 bytes utilised per store sector, and removing the dynamic
   index was worth +3.7% geomean. That line had been sitting in the build log for a
   whole candidate before anyone read it. Reach for it before ncu — **before, not
   instead of.** Read that way it cost the MoE task two candidates: `-Xptxas -v` ruled
   out occupancy, and the next step was a guess rather than a profile. The rule of thumb
   that would have helped: `-Xptxas -v` answers *what the kernel is*, ncu answers *what
   it is waiting for*, and the moment a gap survives the first question, stop guessing at
   the second. In particular `ncu --page source` attributes bank conflicts and stalls to
   individual SASS instructions, which is the only thing that separates two candidate
   causes in the same kernel — on gemm1 it showed the tile stores carrying 92% of the
   excessive wavefronts and the epilogue store, the obvious suspect, carrying 3.8%.
   **Get the stall breakdown before deciding what to fix.** ~~note `--set full` does
   *not* collect it: it needs an explicit
   `--metrics smsp__average_warps_issue_stalled_*_per_issue_active.ratio` run.~~
   **Correction (2026-08-08): that is wrong on this ncu build.** `--set full` *does*
   collect the stall metrics, and they were already sitting in the c6/c7/c8 reports on
   disk — a whole comparison was done from captured reports with no second profiling
   pass and no GPU time. Prefer the `smsp__pcsamp_warps_issue_stalled_*` sample
   **counts** to the `_per_issue_active` ratios: counts are comparable between two
   kernels, ratios move with instruction count as well as latency (rule 9). On gemm1
   that one command reordered the whole plan — global-load latency 32% of the stall
   budget, shared-memory latency 2.5% — and the 2x shared bank conflict that two
   candidates had been aimed at turned out to be real and nearly free. **Wrong is not
   the same as expensive.**
   Read the SASS too, not just the metrics: the winning change came from noticing that
   the k-loop compiled to one register pair reused per load/store, so every shared store
   waited on its own global load.
7. **One variable per candidate.** When a hardware limit forces two at once, record it
   as a caveat rather than hiding it.
8. **A profile shows what the fast kernel does, not what will make yours fast.** Copying
   cuBLAS's tile shape from its kernel name made the demo 20% slower. Treat a metric gap
   as a hypothesis to test. The same holds for a profile of *your own* kernel: it names
   the busiest unit, which is where the work will move **to** as well as where it comes
   from. On the MoE task the busiest unit was L1/TEX at 63%, the fix for it worked —
   the shared-store conflict halved exactly as designed — and the kernel ran 16% slower
   because the cost landed back on L1/TEX and saturated it at 87%. Before making a
   change, size what it *spends* on every unit, not only what it saves on one; and if
   both sides come out the same order of magnitude, that is the argument for measuring,
   not the argument for the change.
9. **Check whether the kernel is issue-limited before spending on instruction count.**
   Twice in one day an instruction reduction bought nothing: a 31.4% cut that made a kernel
   *slower*, and a 4–7% cut that converted to no measurable time at all. The diagnostic is
   cheap and it is in every profile — **eligible warps per scheduler**. At 0.54 eligible of
   8, the scheduler already has nothing to issue and handing it fewer instructions to not-
   issue changes nothing; an instruction cut there is free and worthless in equal measure.
   Spend on instruction count only when the issue pipe is the constraint, which the
   ALU-throughput-bound case genuinely was (+10% for a 2.24x instruction reduction).
   **And when you do spend, size it by the removed instructions' share of the STALL budget,
   not their share of the instruction count.** Two candidates on one kernel give the law:
   one removed **4.1% of instructions carrying 27.6% of stalls** and returned **27.2%**;
   the other removed **12.0% carrying ~2.5%** and returned **2.7%**. Conversion ratios
   0.57 and 0.226 against instruction count — but ~1.0 against stall share, both times.
   A bigger instruction cut is worth less if it comes from instructions nobody was waiting
   on, and `--page source` tells you which those are before you write anything.
   **When a per-instruction rate moves the "wrong" way, close the identity.** Warp cycles
   per issued instruction *rose 42%* (7.60 → 10.78) on a change that made the kernel **27%
   faster**, because 48% of the instructions had left. Do not stop at "the ratio is
   misleading" — multiply it out: 1,463,355 × 7.60 = 11.12M warp-cycles against
   759,754 × 10.78 = 8.19M, a −26.4% that reconciles with the −27.2% duration. If the
   product does *not* reconcile, something else moved and you have not explained your own
   result yet. This is what turns rule 9 from a caution into a test.
   **What a stall table IS good for: confirming a regime, by its direction.** On the grid
   task, removing non-tensor work made `math_pipe_throttle` rise in absolute samples
   (307,581 → 316,516) while *every other row fell*. That is exactly what must happen when
   you remove non-tensor work from a **tensor-throttled** kernel — the freed issue slots
   convert into more time queued on the busy pipe — and it is the opposite of what a
   latency-bound kernel would show. So the table confirmed the diagnosis the same candidate
   depended on, in the same measurement. Use it to answer *what is this kernel*, never to
   score *is this change good*.
   The rest of the original rule still stands: **a stall ratio is not a runtime budget.** `smsp__average_warps_issue_stalled_*` are
   per-*issue* averages, so they move with instruction count as well as with latency.
   The change that won on the MoE task made its own headline metric worse — 6% faster
   with long_scoreboard 7.80 -> 9.24 cycles and warp cycles per issue 24.3 -> 25.97 —
   because it issues fewer, denser instructions that each wait longer and in total wait
   less. Rank candidates by wall time; use the ratios to generate hypotheses, never to
   score them.
10. **Order of work:** harness → reference validated against ground truth →
   broken-kernel self-test passing → *then* the first candidate.
11. **A candidate can win while its hypothesis loses — and worse, it can be right on the
   metric AND the magnitude while its mechanism belongs to something else. Record both, and
   build a control.** The strongest case on this deck: a candidate was pre-registered to
   work by cutting a specific stall row worth 7.86% of the budget. The row fell by half —
   **in the control arm, which changed not one sector of the thing the hypothesis was
   about** — and then *rose* when the actual fix was applied. The predicted size band was
   also hit. Metric confirmed, magnitude confirmed, causal path wrong, and **nothing short
   of a control candidate could have separated them.** A prediction that names only a
   metric and a size is confirmable by accident; only an intervention that isolates the
   mechanism is not.
   The original rule stands: **a candidate can win while its hypothesis loses. Record
   both.** c10 was written
   off the top row of a stall table — `long_scoreboard` at 36% on gemm1 at T=4096 —
   and it is the champion at +1.2%. T=4096 is the one prefill shape it made
   *slower*, on both the median and `ms_min`. The score is real; the mechanism is
   not established, and writing it up as "prefetching hid the global latency" would
   have handed the next round a false premise to build on. Two rounds have now
   aimed at that same stall row: c8 won there, c10 lost there and won everywhere
   else. **A promotion note must say what the candidate was aimed at, not only what
   it scored** — otherwise the ledger accumulates confirmations that were never
   tested. This is rule 9 applied to itself.
12. **In a paired A/B, any shape whose code is unchanged is a free noise-floor
   calibration.** c10 touches only the tile kernels, so T=1 runs byte-for-byte
   identical code in both arms — and swung ±2–3% between them. That number is a
   direct read of the instrument in the same run, under the same clocks and the
   same ambient load, and it says that anything under ~3% on that shape is not a
   result. Cost: nothing. Build candidates so that at least one graded shape is
   untouched, and read it as the control.
   Corollary: **absolute scores are not comparable across sessions on this card.**
   c8 measured 0.1142 over 7 reps in one session and 0.1210 in the A/B that c10
   won — the difference is which clock bin T=512 drew, nothing else. Comparing a
   challenger against a *stored* champion number would have claimed +7.2% here
   instead of +1.2%. Only the paired number means anything.
13. **A validator that can fail is not enough — it must fail for the reason being
   scored.** Rule 4 gets a broken kernel rejected; it does not check that the gate
   exercises the regime the benchmark grades. On the NSA task `check.py`'s two
   shapes hold 4 and 6 blocks against a top-8 selection, so it keeps every block:
   at the graded correctness gate the semantics are plain dense causal attention,
   while the shapes where 8 of 64 blocks survive are never correctness-checked. A
   solution that drops block selection entirely is both faster and PASSing.
   **Before trusting a gate, compute what it actually constrains at its own
   shapes** — and if there is a gap, close it with a task-local probe (`smoke.py`
   at `S >= 545` here) rather than relying on the bench. The same check found the
   grid-env task's `if hit.any()` LCG gate to be invisible to its grader.
   And validate the probe in *both* directions: a gate that fails everything is as
   useless as one that passes everything. Run it against a known-wrong and a
   known-right implementation and record the separation (205.5 vs 0.009 here).
14. **Measure the instrument with a null A/B before trusting a small win.** Run the
   harness with the *same candidate in both arms* — the true difference is zero by
   construction, so whatever it reports is the noise floor. On the MoE task this cost
   178 s of lease and is the most reusable number the task has produced: at 4 reps the
   **geomean floor is 2.6%**, which means the +1.2% geomean that promoted c10 was
   *inside the floor*. The promotion survived only on its per-shape deltas, and one of
   them — T=1000 at +2.65% against that shape's 2.14% floor — should never have counted
   as a win.
   Three things fall out, and they generalise past this deck:
   - **Rank on per-shape deltas against per-shape floors. Treat the geomean as the
     bench's score, not as the experiment's statistic.** A geomean is a weighted product
     of six noise sources; one loose shape poisons it. T=1's 11.4% median swing alone
     contributes ~1.9% of the 2.6%.
   - **Per-shape floors are wildly unequal, and not in the order you would guess.**
     T=4096, the largest shape, was the *tightest* at ±0.19%; the small shapes were the
     loose ones. So a 0.7% regression at T=4096 is three times its floor and real, while
     a 2% "win" at T=1000 is nothing.
   - **Where median and `ms_min` disagree, prefer `ms_min`** — on the noisiest shape it
     was ~4x tighter (2.6% against 11.4%).
   Also learned here: a profile and a sweep can disagree and both be right. `ncu`
   profiles one launch, serialized, with L2 flushed before every replay pass; the
   benchmark runs warm and back-to-back. Any L2 hit rate from a profile is a cold-cache
   number and is not what the benchmark sees.
15. **An effect counts when it replicates in a second independent paired A/B.** Rule 14
   is necessary and not sufficient: a single null A/B measures **one draw** of the
   error, not its spread, so its floors are a *lower bound* on the error rather than an
   estimate of it. Measured on the MoE task: the same candidate pair read
   T=4096 `ms_min` **+0.71%** in one paired 4-rep A/B and **−0.11%** in the next — a
   0.82-point swing against a within-session null floor of 0.23%. A win three times its
   floor in run 1 was simply gone in run 2. Practical reproducibility of a 4-rep paired
   delta is **±0.4 to 0.8 points** at the prefill shapes.
   Replication is cheap next to what it prevents: it costs one more A/B (~3 min of
   lease) and it is the only thing that separates a real 1% from a lucky one. Both of
   this deck's promotion errors — the one rule 5 caught and the one rule 14 caught —
   would also have been caught here, and running it retrospectively *strengthened* the
   c10 promotion (four winning shapes, not three) while shrinking the regression it was
   charged with from −0.70% to −0.25%. Replication corrects in both directions.
   Corollary for harnesses: **make repeated runs the default, not an overwrite.** The
   A/B script keyed its output directory on the candidate pair alone and destroyed the
   first run's logs when the second ran — the same bug class as the one already fixed in
   the repeat script, in the file written beside it.
   **And noise floors do not transfer between tasks — measure your own.** The NSA task's
   instrument is **20–50x quieter** than the MoE task's: worst per-shape floor 0.665%
   against 11.4%. The mechanism is shape size, not kernel quality — the MoE task's noisy
   shapes are tiny decode ones running for microseconds where launch jitter dominates,
   while the NSA task's *smallest* shape still runs 340 µs. A floor borrowed from another
   task is worse than no floor: it carries false confidence in both directions.
   **But the floor does not predict the replication swing, and that is the number that
   decides.** The NSA task's floor is 20–50x tighter than the MoE task's and its
   between-run swing is the *same* 0.4–0.6 points. So the bar is **max(floor, swing)**,
   and the swing only exists if you replicate. A single null draw is also just a draw:
   one shape read 0.000% on both statistics in draw 1 and 0.593% in draw 2, so take
   floors as **max over draws**. Any claim built on one draw — including "this is the
   steadiest shape" — is an over-read.
16. **A promotion harness must not be biased toward promoting, and this one was.** The
   A/B script's arm A, rep 1 only, first three shapes, ran **~32% slow** — the boost ramp
   inside the first ~250 ms of the lease, decaying through the sweep. It appeared in 3 of
   4 runs and *only when the GPU was idle before the lease*; the run that queued 163 s
   behind another worker was clean. Since arm A is always the champion, the bias ran
   systematically **against** the champion — precisely the direction that manufactures
   false promotions. Rule 5 says warm the clocks before measuring; a lease makes it easy
   to forget that the *lease itself* starts cold. Two fixes, both cheap: a discarded
   warm-up inside the lease before the first measured rep, and alternating which arm runs
   first so any residual position effect cancels across reps instead of accumulating on
   one arm.
   Median and `ms_min` at 4 reps rejected this completely; `ms_max` and any mean did not.
   That is a second reason those are the two statistics to report — they are robust to a
   contaminated first rep, which is exactly when contamination happens.
   General form: **audit your instrument for bias in the direction of the result you
   want.** Noise you can average away; a systematic tilt toward the outcome you are
   hoping for survives every rep you add.
17. **Choose the statistic that replicates, not the one that looks small — and check
   that its own error bar replicates too.** On the decode task, `ms_min`'s envelope was
   ±0.5% in both null draws (worst shape 0.49% then 0.47%), while the median's was
   ±2.7% in draw 1 and ±0.7% in draw 2. **A statistic whose own error bar moves 4x
   between draws cannot rank candidates**, however tight it happens to look on the draw
   you took. Report the median because the bench scores it; *rank* on the statistic that
   holds still.
   This has a harness consequence worth planning for: the graded `benchmark.py` may only
   emit the statistic you cannot use, and be forbidden to edit. Then the honest structure
   is a **two-instrument** one — rank candidates on a cheap paired probe that reports the
   reproducible statistic, and spend one graded sweep on the winner for the official
   number. On that task the probe is 2 minutes per pair against 52 for the graded sweep,
   because the sweep pays 7 untimed prefills per shape. **The cheap instrument being the
   reproducible one is not the usual trade** — when it happens, take it, but validate the
   probe against the graded sweep on a pair whose ordering is already known, or you have
   simply moved your trust somewhere unaudited.
18. **Make the hypothesis predict an ordering across shapes, not just a number.** A
   candidate that predicts "this will be faster" is confirmed by any win, including a
   lucky one. A candidate that predicts *which shapes benefit and in what order* can only
   be confirmed by the pattern it named. On the NSA task the mechanism only applies where
   `NW=1`, and every `NW=1` shape (12.1/16.0/9.6%) beat every `NW=2` shape (6.7/4.5/3.2%)
   in **both** independent runs, with the size inside each group tracking that shape's
   share of the kernel. Nothing but the claimed mechanism produces that pattern — which
   is a far harder test to pass by luck than any single delta, and it is free, because
   the sweep measures every shape anyway.
   This is the practical answer to rule 11 (a candidate can win while its hypothesis
   loses): an ordering prediction tells the two apart *in the same measurement*, instead
   of leaving the mechanism unconfirmed until someone spends a profile on it.
   Two corollaries from the same candidate:
   - **The headline stall metric can move the wrong way on a win.** Warp cycles per
     issued instruction got *worse* (9.54 → 12.29) while the kernel got 41.6% faster.
     Ranking on it would have rejected a +10% change. Rule 9, textbook.
   - **Static SASS instruction count is not the measurement.** The cubin *grew*
     (1400 → 1520) because ptxas unrolls further once the body is small. Count dynamic
     instructions from a profile, not lines in a disassembly.
19. **Emulate a numerics change before you build it — and prove the emulator exact.**
   Whether fp16 tensor cores were viable on the grid task was a multi-cycle question
   gating a large rewrite. It was answered in **8.5 seconds of lease with no kernel at
   all**, by a torch emulator, because `wmma` fp16×fp16→fp32 is *exactly* reproducible as
   an fp32 GEMM over fp16-rounded operands: the product of two fp16 values is exact in
   fp32 (11+11 = 22 mantissa bits < 24). The only residue is accumulation **order**,
   which a previous cycle had already priced at 4.66e-9 against the 1e-6 effect under
   test. That separation is what makes the emulator evidence rather than an
   approximation — **do not skip proving it**, or you have built a cheap instrument that
   measures something else.
   The payoff is resolution as much as speed: the emulator ran **1.08 million argmaxes
   against the gate's 3,072**, 350x, which is the only way to measure a 2.86e-6 event at
   all.
   And it changed the answer. fp16 was expected to die on flipped argmaxes at an
   estimated ~3% failure rate; it dies **deterministically on tolerance** instead
   (4.165e-6 against a 1e-6 atol), while the real argmax rate is ~0.9% — an order below
   the estimate. **The estimate had the right order for the wrong reason, and its
   mechanism would not have been what killed the idea.** An estimate whose mechanism you
   have not tested can be numerically close and still send you at the wrong target.
   **The boundary of this technique, which a later cycle found: an emulator running in a
   wider format cannot test format-specific behaviour.** `sm_75` `wmma` turns out to keep
   fp16 **subnormal** inputs exactly, down to the 5.96e-8 minimum — and **98.45% of that
   task's residual values are fp16 subnormal**, i.e. the emulator was blind to precisely
   the regime the data lives in, because an fp16 subnormal is an ordinary normal in fp32
   and torch cannot represent the distinction. Emulate to *rank* and to *kill*; before
   committing to a design, list what the emulation cannot see — subnormals, flush-to-zero,
   fragment layouts, rounding mode of the accumulate — and answer those on the device. On
   that task four such questions were listed and answered in one pass, and one of them was
   the last unknown in the design.
20. **A conflict's cost is not bounded by the stall row naming its pipe, when a barrier
   sits downstream of it.** The MoE task pre-registered a *ceiling* of ~3% for an XOR
   swizzle, pricing a 20% shared-wavefront reduction against the shared pipe's 23.1% of
   the stall budget. It measured **6.6%**, more than double its own ceiling. The reason:
   the tile store is immediately followed by `__syncthreads`, so the conflicted `STS` sits
   on the critical path *into* the barrier — and `barrier` was another 25.9%. A store-side
   conflict in a store-then-barrier loop is priced against **both** rows, ~43% of the
   budget, because every extra bank cycle before the barrier is paid by all 16 warps.
   This also retro-explains an earlier candidate that halved the *same* conflict and lost
   16%: it paid for the halving on the global side, so it never collected the
   barrier-side saving. Same conflict, opposite outcomes, and the difference is which side
   of the barrier the cost landed on.
   **Build a control candidate to attribute a two-part change.** The winning change had
   two components; an intermediate was built with only one, purely to decompose it —
   never promotable, never intended as a candidate. Result: instruction count +0.83,
   swizzle +6.26, both together +6.0–6.6, additive to within the reproducibility band and
   with the same ordering at every prefill shape. Without it the ledger would have said
   "swizzle plus ldmatrix is worth 6.6%" and the next worker would have inherited a claim
   about the wrong half.
   Corollary calibration from that control: halving a k-loop's `ldmatrix` count at
   identical wavefronts — 24 `LDSM.M88.2` → 12 `LDSM.M88.4`, HMMA density 38.6% → 47.4% —
   was worth only ~0.9%. On a kernel like this, **issue slots are nearly free; what costs
   time is what the memory pipe actually does.**
   And the backlog lesson: this hypothesis had been ranked *last*, on a falsifier ("the
   conflict is real and nearly free") that a later profile retired. It then beat
   everything ranked above it combined. **When a profile retires a hypothesis's stated
   falsifier, re-rank it — do not leave it where its dead objection put it.**
21. **Design a null arm into the experiment: a shape the mechanism predicts will NOT
   move.** Rule 18 says predict an ordering; this is its sharpest form. The NSA task's
   +11.5% change reduced shared memory per block, so it should help only where the saving
   crosses a *block-count* threshold. One graded shape received the **identical** 5120-byte
   saving and could not cross the threshold — and it was the only shape that did not move,
   in both runs, on both statistics, while every other cleared its floor by 16–20x. That
   single flat shape is stronger evidence than the five wins: it separates "the block
   count is the mechanism" from "the byte count is the mechanism," which no amount of
   winning could. **It also cost nothing** — the null arm was already in the graded sweep,
   waiting to be recognised.
   State in advance which shape is the null and why. A null arm found after the fact is
   an observation; one named before is a test.
   **The null arm can be a source line, not just a shape.** On the grid task a change fixed
   one bank-conflict site among five in the same kernel. The time saving landed on exactly
   the two stall rows a conflict occupies (72.6% of the whole drop between them), and the
   **four untreated sites were unchanged to the digit** — a control at source-line
   granularity, already sitting in the profile, requiring no extra run. Whenever a change
   treats one instance of a repeated defect, the untreated instances are the control.
   **A null arm also calibrates the profiler, not just the sweep — and the number differs
   within a session and across them.** The decode task measured ±0.27% for a byte-identical
   kernel re-profiled *within* one session; the grid task, re-profiling a byte-identical
   binary in a *different* session (same `inst_executed` to the digit), measured **±1.6%**.
   **Counts reproduce exactly; durations do not.** Quote kernel durations only against a
   measurement from the same session. On the decode task the untouched kernel re-profiled
   36.448 → 36.355 µs
   in a different session — so a per-kernel duration on that box is good to **±0.27%**, and
   the two treated kernels moved 6x and 28x that. The task had no per-kernel calibration
   before, and this cost nothing because the null arm had already been named for a
   different reason. A profiled duration quoted without one of these is an unbounded number.
   And report what did *not* resolve: on the same candidate the within-group ordering came
   out 2>1>5>0>4 in one run and 5>1>2>0>4 in the other, with predicted shares of 84/75/73%
   too close for the instrument. The grouping replicated; the fine order did not. Claim
   the grouping.
   **When a win is concentrated on one shape, make that the headline and justify the fast
   path as a property of the INPUT.** A +17.4% geomean whose entire gain is one of six
   shapes — +165% there, nothing elsewhere — is legitimate only if the fast path triggers
   on something the input actually *is*: here a runtime `nb <= top_n_blocks` test making the
   selection provably a no-op, exact rather than approximate, with bit-identical output over
   1.2M values. The same +17.4% obtained by branching on a shape from the benchmark's own
   table would be a shape table with extra steps. **The test is whether the condition would
   fire on a workload the deck never scores.** State the concentration as the headline, not
   as a footnote — a geomean is a geomean, and a reader who has to derive that for
   themselves has been misled by omission.
22. **A tight fit to aggregate data can be entirely artefactual. Decompose before you
   trust a model.** The decode task fitted `t = 93.9 µs + traffic/343.4 GB/s` across four
   shapes to **under 1.7% residuals**, and used it to size two levers and rank a backlog.
   A profile then showed **every term is a mixture**: at the long shape the step is 1315 µs
   of attention at 409 GB/s, plus ~340 µs of weight kernels at 371 GB/s, plus ~66 µs of
   inter-kernel gap. The "streaming rate" is an average over kernels running at different
   rates, and the "fixed cost" is an average over per-kernel overheads. Neither names a
   kernel, so neither can be optimised.
   The damage was concrete: the model blamed the attention kernel for the shortfall, and
   the attention kernel turns out to be the **fastest-streaming thing on the task** at
   95.84% of the DRAM roof. A cycle was spent on a candidate aimed at it.
   **Residuals measure how well a curve fits, never whether its parameters mean anything.**
   With four points and two free parameters, a good fit is nearly guaranteed. Before
   trusting an aggregate model, profile once and check that each term corresponds to
   something you could actually change.
23. **A profiler's own flagged metric is a hypothesis, not a diagnosis — and it can be
   driven to its ideal value for nothing.** `ncu` flagged `Waves Per SM = 2.67` on the
   decode task's weight kernels and named the tail loss itself; the arithmetic predicted
   11.1% and agreed with the measured bandwidth to 0.8%, which is exactly the kind of
   agreement that feels like confirmation. A candidate took that metric to **1.00 waves**
   — the ideal, on the exact kernel named, with achieved occupancy going 88% → 98% — and
   the kernel got **0.7% slower**. The model predicts 100% efficiency at 1.00 waves;
   measured 82.8%. A 17-point miss on a direct test.
   Two things this buys. It is the cleanest way to kill a model: **drive its metric to the
   ideal and see whether the time follows.** If it does not, the metric was correlated
   with the cost, not causing it — and no amount of further tuning along that axis will
   help. And it is another instance of arithmetic agreeing with a measurement without
   either being the mechanism (rule 22): two numbers matching to 0.8% proved only that
   both were downstream of something else.
   What the same profile *did* order correctly was a ratio nobody had flagged — the share
   of a kernel's traffic spent staging into shared memory. Four kernels, one ordering:
   stage nothing → 98.6% of the roof, 1:8 → 83.4%, 1:4 → 69.6% and 67.4%. **When a
   flagged metric fails a direct test, look for the quantity that orders *all* the
   kernels, not just the one that was flagged.**
24. **Before trusting a spread, check how much of it is the instrument — a bad instrument
   inverts rankings, and rankings are what hypotheses are built on.** The decode task
   measured five streaming kernels spanning **67.4–98.6% of the roof, a 31-point band**,
   and every hypothesis on its page was reasoned from that spread. Under a corrected
   measurement the same five kernels land in a **twelve-point band**, 62.7–75.4% — and the
   kernel previously called *the fastest-streaming thing on the task*, which had been used
   to **kill a hypothesis**, is actually the **slowest** of the five. That hypothesis's
   death was an artefact and had to be retracted.
   *This example was first published as a six-point band, from a column that summed reads
   and writes and so carried ~1.4 MB/launch of phantom counter — and it went into this
   playbook that way. The rule needed applying to its own worked example, and the error ran
   in the flattering direction, since a tighter band makes the corrected instrument look
   better. The conclusion survives (12 points is still far below the 31 the artefact
   produced), but a rule about instrument error being stated from a contaminated
   measurement is the most on-the-nose entry in this document, and it is left visible
   rather than quietly fixed.*
   It also explained a pattern two cycles had been chasing: interventions that landed
   perfectly and recovered 12.8% and −17% of their predicted gains were all sized against
   denominators from the same bad instrument. **When several well-executed changes all
   under-deliver by large and varying factors, suspect the denominator before the ideas.**
   Corollary, from the same cycle: **a microbenchmark that isolates a cost tells you what
   that cost is when nothing else is happening.** A no-op launch schedule measured "66 µs
   per step of inter-kernel gap"; in a real kernel that grid ramp is inside the kernel's
   own duration, the 29 real kernels sum to the whole step, and the gap is bounded above by
   **zero**. An entire lever — a CUDA graph over the schedule — was dead on arrival. Same
   error as an artefactual aggregate fit (rule 22), one level down.
25. **Theoretical occupancy is a property of the kernel; achieved occupancy over time is a
   property of the *schedule*. Ask both.** On the NSA task three candidates raised
   *theoretical* occupancy from 25% to 37.5% by cutting shared memory, worth +11.5%, +3.8%
   and +2.3%. A fourth raised *achieved* occupancy, changed nothing else, and was worth
   **+10.4%/+10.6% on all six shapes** — more than the other three combined. Six cycles of
   `ncu` had asked how many blocks fit on an SM and none had asked whether the SMs were
   full.
   **The specific trap: block cost linear in the block index.** That kernel's block `qt`
   loops `kb = 0..qt`, so cost runs one key-block to sixty-four. The grid made the linear
   id `qt + ntiles*bh`, and the hardware issues blocks in increasing linear id as slots
   free — so it ran each `(b,h)` **cheap-first, eight times over, with the single most
   expensive block issued last.** That is the textbook worst case for list scheduling: the
   longest job starts with nothing left to overlap it, so its entire duration is tail. The
   fix was two lines, swapping the grid dimensions and reversing the index to issue
   expensive-first.
   **Whenever block cost varies systematically with the block index — causal masks,
   triangular loops, variable-length sequences, anything `for (j = 0; j <= i; ++j)` —
   check the issue order.** It costs nothing to look, the fix is usually a grid reshape,
   and no resource-footprint metric will ever show it to you.
26. **A stall budget is a budget for one kernel in isolation. It cannot price what a kernel
   does to the machine that the *next* kernel pays for.** On the MoE task `barrier` was the
   largest row at **28.65%** of the stall budget — and halving it was worth **0.0%**. Not
   for rule 23's reason: the metric moved, the saving was real, and it was collected inside
   the kernel. `gemm1` alone went 14.512 → 13.750 ms, −5.3%, with barrier samples −16.1%.
   **The forward got slower anyway.** Both numbers are true, and the difference is a
   whole-forward cache effect that a single-launch, L2-flushed profile is structurally
   unable to observe.
   **But it is a two-sided TEST, not only a warning — and that is the useful half.** The
   two counters that explained the loss also *predict* the win. When a per-kernel profile
   shows a gain, check what the kernel did to **L2 traffic** (`lts__t_sectors`) and to the
   **shared carveout** (`launch__shared_mem_config_size`). The losing candidate had
   `lts__t_sectors` **+9.4%** and a carveout change; a later one had **−0.3%** and no
   carveout change, and its isolated −14.1% duly transferred — 5.7% of the forward
   predicted against +4.65/+5.13% measured, agreeing to within a point. **So you can tell
   which story you are in before spending the A/B.**
   So a per-kernel profile answers "what is this kernel waiting on" and never "what will
   this change cost the step". When a kernel-level win and a step-level loss disagree,
   **do not pick the one you like** — the profile is not wrong, it is answering a smaller
   question. Attribute the difference before spending another candidate.
   The concrete instance is worth carrying on its own: **declaring shared memory you never
   touch cost 2%** — ten times that shape's noise floor. On sm_75 the 96 KB of L1/shared
   carves into 32 KB or 64 KB of shared, so a block declaring **more than 16384 B** forces
   the 64 KB carveout and leaves only 32 KB of L1. The declaration alone, with the memory
   unused, halves your L1. Check the carveout before adding a shared buffer "just in case",
   and note the tempting explanation — a carveout mismatch between two kernels forcing a
   per-launch reconfiguration — was tested here and **falsified**, so do not reach for it.
27. **Excessive shared-memory wavefronts are a defect count, not a currency.** The grid task
   removed **7,340,032** excessive wavefronts and gained +1.9–2.1%. It then removed
   **3,670,016** more, at the *same sites*, in the *same kernel*, at the *same shape* — and
   gained **nothing**: a control candidate isolating that fix could not clear a single one
   of its eight per-shape floors in either direction. Half the defects for zero percent.
   They are not on one line through the origin.
   **A conflict's price is not a property of the conflict.** The MoE task carried the *same*
   XOR swizzle into a second kernel — the same 46–47M store conflicts, in the same
   store-then-barrier loop, 98% as large — and it was worth **6.3% in one kernel and 3.4%
   in the other**. Both directions were predicted in advance from context: the second kernel
   runs 31.4 active warps against 15.9, so more of a conflicted store is covered by other
   warps, and its `long_scoreboard` was already 23.85% against 5.11%, so more of the saving
   converts into re-exposed global latency rather than into time. Two tasks reached this
   from opposite directions on the same day.
   **The variable is n-way, not the count.** The first site was **n-way 16**: one `STS` held
   the LSU sixteen bank cycles, stalling both its own dependents and the pipe behind it —
   two stall rows carrying 72.6% of that candidate's entire time drop. The second sites were
   **n-way 2**: one extra bank cycle, spread over 4x the instructions, on a shared pipe only
   **7.7% busy**. Their stall rows moved by 16% of the drop.
   So: **price a conflict by what it does to the instruction that owns it — n-way, and
   whether the pipe is busy — never by the aggregate excess it contributes.** The
   **And price it against the whole workload, not the kernel.** A frontier verdict needs a
   divisor: one kernel was 34% of its forward, so a change confined to it had to be worth
   ~2% *of that kernel* merely to clear the replication band on the score. The largest
   remaining defect there — n-way 8 on a pipe 88.6% busy, the most favourable corner
   available on both variables of this rule — was worth 1.99% of the kernel and **+0.3% of
   the forward**. Once what remains is smaller than what was just taken, every unwritten
   idea is disqualified by arithmetic, and that is what `frontier-thin` should mean.
   The consequence was immediate on that task: a queued hypothesis holding 2,621,440 excess at
   n-way 4 was re-sized *down* and dropped before anyone built it, saving a cycle and three
   coupled variables.
   This is also the strongest argument in this document for **control candidates** (rule
   20). The promoted change had two components, only one pre-registered; the control showed
   the pre-registered one was worth approximately nothing and the win was an incidental
   1.79% instruction cut that arrived when the loop rewrite removed spills. Without the
   control the ledger would have recorded a confirmed mechanism that was not the mechanism.

28. **An instrument that cannot honour its argument must exit, never fall back.** The shared
   `prof_driver.py` template selected a shape with `reference.T = ARG` — problem 03's own
   knob, generalised by accident. On every other problem that line sets an attribute nothing
   reads, so the driver profiled the **default** shape whatever you passed, and the
   `.ncu-rep` looked entirely normal: right kernel, right occupancy, plausible stalls, wrong
   workload. Task 02 lost every profile it took before cycle 2 and only found out because it
   cross-read a duration against its own phase timing. Nothing downstream could have caught
   it — a profile carries no record of what it was *asked* for.
   Two properties make a selector safe, and a driver needs both. **Validate against the real
   list and exit on a miss** (`if idx not in graded: sys.exit(...)`) — a silent substitution
   is worse than a crash, because a crash costs a minute and a substitution costs every
   conclusion drawn from the report. **Copy every field of the selected shape, not the one
   that varies today** — a driver that copies only `T` goes wrong the first time a deck
   varies a second field, and that failure is silent too.
   The generalisation is not about profilers. Any harness that takes a *selector* — a shape
   index, a candidate id, a seed, a graded-case name — is making a claim about what it ran,
   and that claim is the only thing tying its output to a question. Ask for a run whose
   name it cannot honour and it must refuse. Corollary: **before trusting an instrument's
   output, check the instrument reads its own argument** — pass a deliberately invalid
   selector and confirm it fails. That is rule 4 applied to tooling instead of kernels.

## This machine

```bash
/usr/local/cuda/bin/nvcc   # CUDA 13.3 — not on PATH, use the absolute path
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -lcublas src/main.cu -o runs/bin
```

- **RTX 2070**, TU106, **sm_75**, 36 SMs, 8 GB, idles at 315 MHz, max 2160 MHz.
  It also drives the display, so expect a few % of run-to-run drift.
- **Static shared memory caps at 48 KB per block** on sm_75. More (up to 64 KB) requires
  dynamic shared memory plus `cudaFuncSetAttribute`.
- **Nsight Compute works** on WSL2. Two traps: always pass `-k "regex:<kernel>"` or
  `--launch-count 1` captures whatever launched first (often a warmup or library kernel)
  instead of yours; and `--set full` replays ~34 passes, so profile a single launch.
  Pair `--profile-from-start off` with a `torch.cuda.profiler.start()` after warmup to
  guarantee the captured launch is a warm one — see
  [swarm/harness/profile.sh](swarm/harness/profile.sh) and its
  `prof_driver.py`. Namespace the output directory by candidate: profiling a second
  candidate into the same path silently overwrites the first's report.
- Read metrics back without the GUI: `ncu --import x.ncu-rep --page details` and
  `--page raw` (grep for `sm__pipe_tensor_cycles_active`,
  `l1tex__data_bank_conflicts_*`, `dram__bytes*`). **`--page source --csv` is the one
  worth knowing**: it gives per-SASS-instruction `L1 Wavefronts Shared Excessive`,
  `L1 Conflicts Shared N-Way` and stall sampling, which is what turns "8.4-way bank
  conflict somewhere in this kernel" into a specific `STS.128`.
- **Use UNSIGNED integers for an LCG or any wrapping arithmetic.** Signed overflow is
  undefined behaviour in C++, and `nvcc -O3` exploits it: on the grid task a signed
  `int64` LCG had its second mask **deleted by the optimiser**, producing a kernel that
  failed the correctness gate for reasons invisible in the source. `unsigned long long`
  with the same mask is well-defined and matches torch's int64 wrap bit-for-bit.
- **`nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi`,** not on the default root PATH.
  Clock locking does not work regardless: `-lgc` returns `Unknown Error` against the
  Windows driver. See rule 5.
- **The 59.7 TFLOP/s fp16 tensor peak is the fp16-ACCUMULATE rate, and most kernels do
  not get it.** Measured on this card with register-resident dependency chains:
  wmma `m16n16k16` **f16*f16→f32 = 35.32 TFLOP/s**, f16*f16→f16 = 50.84, fp32 FFMA =
  **8.85**. So an fp32-accumulate tensor kernel — which is almost all of them — is
  bounded **1.69x below the datasheet**, while the SIMT roof is **1.18x above** its
  quoted 7.5 (that figure is at the nominal 1620 MHz and the card boosts past it). The
  widely repeated "GeForce Turing halves fp32 accumulate" is wrong here too: the
  measured penalty is **1.44x**, not 2x. **CONTESTED — and probably wrong.** A second
  independent measurement gets **71.10** for fp16-accumulate, i.e. exactly the 2x
  textbook ratio, and reproduces 51.89 at *2* accumulator chains rising to 71.1 at 4+.
  The fp32-acc figure replicated across both harnesses (35.62 vs 35.32, 0.8%) and so did
  FFMA (8.72 vs 8.85), so only the fp16 arm is in doubt. The tiebreak is an identity, not
  a re-measurement: **every roof implies a core clock, and they must agree.** fp16-acc
  peak is 36 SM x 1024 FLOP/clk, SIMT fp32 is 2304 x 2, so 50.84 implies 1.379 GHz while
  that same run's FFMA implies 1.921 — a card cannot clock its tensor pipe 28% below its
  FP32 pipe in one session. Treat **2x** as the ratio and 50.84 as ILP-limited until the
  fp16 arm is re-run at greater chain depth.
- **Measure any roof at several dependency-chain depths, and report the plateau.** The
  fp32-acc arm above reads 25.5 / 33.3 / 35.6 / 35.6 TFLOP/s at 2 / 4 / 8 / 12 chains: a
  single-depth measurement records the *dependency stall* as the roof and understates it
  27%. This is exactly how the 50.84 above got recorded. A roof that has not been shown
  to plateau is a lower bound on the roof, not the roof.
  **This moves the machine balance from 144.2 to 85.3 FLOP/byte**, which is enough to
  flip a verdict: on the MoE task two shapes at AI 141.4 and 143.5 read memory-bound
  against the datasheet roof and **compute-bound** against the measured one. Quote the
  measured roof for the accumulate type your kernel actually uses, and recheck any
  "which roof binds" call made against 59.7.
- **Achieved DRAM bandwidth is 405-414 GB/s** on a read-heavy streaming kernel, against
  the 448 GB/s spec peak and the 359 GB/s a memcpy benchmark reports. Use the higher
  figure for bandwidth floors.
- **Everything that touches the bench runs as root**, because the venv interpreter is a
  symlink into root-owned `/root/.local/share/uv`. `run.sh`, `repeat.sh`, `ab.sh`,
  `profile.sh` and `gpu.sh` all re-exec under `sudo -n`. Two consequences that cost
  time when the parallel harness was built:
  - **`sudo` scrubs the environment.** Any state passed between these scripts needs
    `sudo -n --preserve-env=VAR`. The GPU lease is reentrant via `CB_GPU_LEASE`, and
    losing it across a `sudo` makes a script block on a lock it already holds — a
    self-deadlock that presents as an unexplained hang.
  - **`fs.protected_regular = 2`.** A process cannot open for write a file it does not
    own inside a sticky world-writable directory such as `/tmp` — and this applies to
    root as well, so `chmod 666` does not help. A lock or log file in `/tmp` created by
    the calling user is then un-openable by the same script running as root, and the
    only symptom is a bare `Permission denied`. Create such files as one consistent
    user, or keep them outside `/tmp`.
- **`core.fileMode` is `false` in this repo.** `chmod +x` never reaches the index, so a
  new script committed from here checks out non-executable in a `git worktree` and a
  worker's first `./run.sh` is a permission error. Use
  `git update-index --chmod=+x <path>`.

## Skills

KDA's README tells you to link `ncu-report-skill` and `KernelWiki` into
`~/.claude/skills` (not yet done here; skills load at session start, so restart after).
What it does not tell you is that both target newer hardware than this machine:

- **`ncu-report-skill` targets B200/sm_100.** Workflow and per-run directory convention
  transfer to sm_75; its metric names (`08-b200-metric-names.md`) do not.
- **`KernelWiki` targets Blackwell/Hopper.** `tcgen05`, `tmem`, `TMA`, `wgmma`, and
  NVFP4 pages are irrelevant on Turing. Architecture-general pages do transfer:
  `swizzling`, `double-buffering`, `vectorized-loads`, `tile-scheduling`,
  `register-budgeting`, and `patterns/`.

Both can be read directly as reference material without installing them.

## Reusable pieces

[workspace/fp16-gemm/src/main.cu](workspace/fp16-gemm/src/main.cu) is a harness worth
copying: candidate registry, shape-alignment guards, `median_ms` + `warm_up_clocks`,
`rel_frobenius`, the two-stage check (ground truth → reference → candidates), the
broken-kernel self-test, CSV emission. Task-specific: the kernels, the reference call,
the shape list. [workspace/fp16-gemm/run.sh](workspace/fp16-gemm/run.sh) wraps it as
`build | check | bench | profile <id>`.

When the *bench* owns the harness, the pieces worth having alongside it are smaller and
live in [swarm/harness/](swarm/harness): `run.sh` (install a candidate as
`solution.py`, then run the bench's own `check.py` / `benchmark.py` under the right
toolchain), `repeat.sh` + `collect.py` (rule 5's second half — repeat the run, reduce to
per-shape medians, keep the spread visible), a per-task `smoke.py` (seconds-long
correctness probe so a layout bug shows up before paying for the real gate), and a
per-task traffic model like `01-glm52-fused-moe/tiles.py`, which is what found the
bottleneck there. A roofline's `bytes_formula` describes the problem, not your schedule;
if your kernel re-reads anything, model that separately before writing candidates.

## Deviations from KDA, on purpose

- `docs/report.md` is a **local addition**. KDA's evidence list stops at
  `candidates.jsonl` and `profile/`; it asks for promotion decisions to be recorded, not
  for a narrative file. Useful, but not a KDA requirement.
- The demo did **not** write the `profile/<run>/REPORT.md` that `ncu-report-skill`
  mandates for every profiling run. The MoE task does — see
  [01-glm52-fused-moe/profile/](trials/2026-08-08-t1/01-glm52-fused-moe/profile/), one
  directory per (candidate, kernel, shape). It earns its keep for the runs that
  *rejected* a change: the c6-vs-c7 pair is the whole argument for rule 8's second half,
  and neither `candidates.jsonl` nor the CSV has room for it.

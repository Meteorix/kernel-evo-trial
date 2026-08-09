# cuda-bench — the frozen starting point

Everything a KernelBench-CUDA evolution trial needs to begin, and **nothing a trial
produces**. A trial never writes here. `../evo-trials/<name>/` is where a run lives.

```
cuda-bench/
  harness/     gpu.sh run.sh ab.sh repeat.sh profile.sh wstat.sh timeline.py status.sh
  template/    STATUS.md SWARM.md README.md   — what a task is seeded with
  tasks/       <NN>-<name>/CONTRACT.md   — the contract, transcribed from the bench
  start-trial.sh
```

Start a trial:

```bash
./start-trial.sh 2026-08-09-t2            # all four tasks
./start-trial.sh 2026-08-09-t2 01 03      # a subset
```

## Why the split exists

Trials are only comparable if they start from the same place. A second run that
reaches 75% of a DRAM floor means one thing if the first reached 30% and another if
it reached 76% — and the question worth asking of this system is not whether it
produces a fast kernel once but whether it does so *reliably*. That needs a fixed
starting line.

So this directory is deliberately **minimal and reusable**: the bench's own task
definitions, the harness, and the scaffolding. It does not accumulate. A trial's
tools, work models, hypotheses, candidates, ledgers and profiles all stay in the
trial that produced them.

## What is a contract here, and what is not

`tasks/<NN>/CONTRACT.md` is a **faithful transcription** of the task owner's files —
`PROMPT.txt`, `problem.yaml`, `shapes.py`, `check.py`, `benchmark.py` — plus the
machine context every task needs. Nothing in it is invented (playbook rule 1).

It deliberately does **not** carry the hand-counted work model. That is a *derived*
artefact: executed FLOPs and bytes per shape, arithmetic intensity, which roof binds,
and the resulting floor. Rule 2 says a trial computes it before its first candidate,
and the evidence for putting it there is strong — the two tasks that computed floors
in cycle 1 had their main hypotheses corrected before any GPU time, while the two
that did not spent candidates first and found the number at cycle 5.

Shipping a pre-computed work model would hand every future trial that correction for
free, and then nobody learns whether the loop can derive it. Trial 1's models are
preserved in `../evo-trials/2026-08-08-t1/<task>/CONTRACT.md` for comparison.

## What carries between trials, and what does not

| | carries | why |
| --- | --- | --- |
| `harness/` | yes, as a frozen copy | six harness bugs were found and fixed during trial 1 — a promotion bias in the A/B, a hardcoded regex in the profiler, two clobbering bugs, an exit-code bug, an env-scrubbing deadlock. Re-finding those would measure nothing useful. Fixes made *during* a trial live in that trial until deliberately promoted back here. |
| `template/` | yes | shared ledger schema is what lets the leader compare tasks at all |
| `tasks/*/CONTRACT.md` | yes, transcription only | see above |
| `../KDA.md` (repo root) | **yes, and this is a choice** | accumulation is the point of the system. But it means trial 2 starts with 27 rules where trial 1 started with 10, so the two are *not* a controlled comparison of the loop alone. State in `TRIAL.md` which playbook revision a trial began from. |
| tools, work models, hypotheses, candidates, ledgers, profiles | **no** | these are what a trial is *for* |

The playbook row is the one to think about before running a comparison. If you want
to measure the loop rather than the loop-plus-knowledge, pin the playbook to a
revision in `TRIAL.md` and say so.

## Promoting a fix back

If a trial fixes the harness or the template, copy it here in its own commit, with the
evidence. That commit changes the starting line for every later trial, which is
exactly why it should be deliberate rather than a side effect.

# <NN> <task> — report

The narrative for a human reader. `candidates.jsonl` is the data; this is what
it means.

## Result

| | |
| --- | --- |
| Champion | |
| Score | |
| Baseline | |
| Candidates spent | |

## What worked

Per promotion: the change, the number, and **why** it won — the mechanism, not
the metric. "DRAM throughput rose" is a metric; "the k-loop no longer serialises
each shared store on its own global load" is a mechanism.

## What did not

The rejected candidates, with the same standard. This section is the reason the
ledger is worth keeping: it is what stops the next worker re-buying a closed
frontier.

## Measurement caveats

Anything that makes a number less portable than it looks: bimodal shapes,
ambient load during the run, reps taken, whether it was re-measured quiet.

## Open frontier

Points at `HYPOTHESES.md` rather than duplicating it.

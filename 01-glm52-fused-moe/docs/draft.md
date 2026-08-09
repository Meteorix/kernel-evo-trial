# <NN> <task> — draft

Written **before** the first candidate. Playbook rules 1–3 and 10.

## Contract

Copied from `CONTRACTS.md`, which was itself copied from the task owner's files.
State here any field that is *not* in the contract and that this draft is
inventing — playbook rule 1 exists because the KDA demo invented the shapes, the
tolerance and the target, then reported success against its own target.

## Reference validation

The reference must be checked against a slow, obviously-correct ground truth
(CPU float64, small shape) **before any candidate**. A layout or transpose
mix-up otherwise makes every later number wrong while everything "passes".

- Ground truth used:
- Shapes checked:
- Result:

## Broken-kernel self-test

`candidates/c_broken.py` — a deliberate, specific error, plus enough real CUDA
to satisfy the language gate. `check.py` must **FAIL** it.

- The error introduced:
- `check.py` output:

## Arithmetic floors

What the hardware says is possible, before any kernel exists. DRAM traffic at
405–414 GB/s achieved on this card (not the 359 a memcpy benchmark reports, and
not the 448 spec peak); fp16 tensor peak 59.7 TFLOP/s; fp32 SIMT 7.5.

Knowing the floor is what closed task 01's T=1 frontier without spending a
candidate on it.

## Planned candidates

One variable each. If a hardware limit forces two at once, record it as a caveat
rather than hiding it.

# Task harness template

Copy these into a new `workspace/cuda-bench/<NN>-<name>/` before the first
candidate. They exist so the four tasks stay one deck: three workers building
scaffolding independently would otherwise invent three ledger formats, and the
leader could not compare or merge them.

```bash
mkdir -p <NN>-<name>/{tools,candidates,runs,profile}
cp template/{collect.py,prof_driver.py,smoke.py} <NN>-<name>/tools/
cp template/HYPOTHESES.md <NN>-<name>/
cp -r template/docs <NN>-<name>/
```

**The layout, and it is enforced by convention not by code.** A task directory holds
**six state files** — `CONTRACT.md`, `STATUS.md`, `HYPOTHESES.md`, `candidates.jsonl`,
`cycles.jsonl`, `config.json` — and **five directories**: `tools/`, `candidates/`,
`docs/`, `runs/`, `profile/`. Nothing else at top level.

**Every script you write goes in `tools/`.** This is the rule that matters, because
it is the one that decays: probes and one-off models accumulate fast — one task
reached 30 top-level entries, of which 20 were ad-hoc `*_probe.py`. They are worth
keeping (they are how a later cycle re-checks a claim) but not worth putting in the
way of the six files a worker actually reads first.

If a tool locates itself, remember it now sits one level down:
`Path(__file__).resolve().parent.parent` is the task dir, and in shell
`$(cd "$(dirname "$0")/.." && pwd)`.

`collect.py` is deck-generic and needs no edit. `prof_driver.py` and `smoke.py`
carry `TODO(task)` markers where the problem's own input construction goes —
tasks 03 and 04 expose `run(...)` rather than a `Model.forward`, so their
drivers differ more than 01/02's do.

Optional `<task>/shape_labels.json` maps shape index to a readable label and is
used by `collect.py` only for display:

```json
{"0": "T2048", "1": "T2079", "2": "T1", "3": "T4096", "4": "T512", "5": "T1000"}
```

## Order of work

`KDA-PLAYBOOK.md` rule 10, and it is not negotiable:

1. harness
2. **reference validated against a slow, obviously-correct ground truth** — CPU
   float64, small shape. A layout or transpose mix-up otherwise makes every
   later number wrong while everything "passes"
3. **broken-kernel self-test asserted FAIL** — ship `c_broken.py`, a candidate
   with a deliberate error, and confirm `check.py` rejects it. A validator that
   never fails is worse than none
4. *then* the first real candidate

Tasks 02–04 start in `ramping` and leave it only when 3 is done.

## `candidates.jsonl` — the ledger schema

One JSON object per line, appended, never rewritten. This is the task's memory:
per-round workers have no other way to know what has already been tried, and the
`note` field is where the reasoning lives.

| Field | When | Meaning |
| --- | --- | --- |
| `id` | always | `c0`, `c1`, … or a label for a note row (`profile-stalls`) |
| `parent` | always | candidate this was derived from, or `null` |
| `status` | always | `baseline` \| `promoted` \| `rejected` \| `selftest` \| `note` |
| `change` | always | **one variable.** If hardware forced two, say so here |
| `hypothesis` | candidates | which `HYPOTHESES.md` entry this tests |
| `validated` | candidates | did `check.py` run to completion |
| `check` | candidates | `PASS` / the failure text |
| `framework` | candidates | from the language gate; must not be `pytorch_only` |
| `geomean` | candidates | the score |
| `ms` / `tok_s` / `sps` | candidates | per-shape medians, task-appropriate metric |
| `ms_min` | candidates | per-shape minima — **required**; see rule 5 |
| `reps` | candidates | how many; `>= 3` for a promotion |
| `ambient` | candidates | `{"agents_active": N, "loadavg": X}` at measurement time |
| `regs` / `smem_bytes` | when known | from `nvcc -Xptxas -v`, per kernel |
| `note` | always | **why it won or lost.** The most valuable field in the file |

A rejected candidate with a good `note` is worth more than a promoted one
without. Task 01's `c7` (−16%) and `c9` (−15%) are both rejected and both among
the most useful rows in its ledger.

## Measuring

- `../ab.sh <NN> <champ> <chal> <reps>` — the promotion decision. Interleaves
  both arms inside one GPU lease so they see the same clocks and the same
  ambient load.
- `../repeat.sh <NN> <cand> <reps>` — characterise one candidate (spread,
  bimodality, a baseline).
- `../profile.sh <NN> <kernel-regex> <shape-arg>` — Nsight Compute, one warm
  launch.
- **Rank on `ms_min` as well as the median.** Rule 5's second half: a median over
  reps of a bimodal shape reports which clock bin came up more often.

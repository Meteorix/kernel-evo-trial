# <NN> <task> — hypotheses

The task's backlog and its closed frontier. Workers are spawned per round and
carry no memory of each other, so **this file is the memory**. A worker reads it
before choosing what to build and updates it before exiting.

Closing is as valuable as opening. A hypothesis closed with evidence is progress
and resets the stall counter (see `PARALLEL-PLAN.md` §4.3 at the repo root) — task 01's `c7` and
`c9` were both large regressions and both among the most useful rounds in its
ledger.

---

## Open

Ordered by expected value. Each entry:

### H<n> — <one-line statement>

- **Source:** profile / SASS / arithmetic floor / playbook rule / contract
- **Predicts:** what changes, on which shapes, by roughly how much
- **Cost:** rough — a tweak, a candidate, or a rewrite spanning rounds
- **Falsified by:** the measurement that would kill it
- **Notes:** anything a fresh worker needs in order to not restart the thinking

---

## Closed

Never delete an entry; a closed hypothesis is what stops the next worker
re-spending a candidate on it.

### H<n> — <statement> — CLOSED by <candidate or measurement>

- **Result:** what happened, with numbers
- **Why it is closed:** the reasoning, not just the outcome

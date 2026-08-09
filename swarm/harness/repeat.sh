#!/usr/bin/env bash
# Run the bench's benchmark.py N times for one candidate, keeping every run.
#
#   ./repeat.sh 01 c2_bm128 3
#
# Why this exists: benchmark.py already warms and medians *within* a run
# (src/eval/timing.py), which is enough for the prefill shapes but not for the
# small ones. Re-running c2 unchanged moved T=1 by 37% and T=512 by 14% —
# wider than the gap between the candidates being compared. The card drives the
# display and idles at 300 MHz, and `nvidia-smi -lgc` cannot lock clocks under
# WSL2 ("Unknown Error" against the Windows driver), so the only honest fix is
# to repeat the run and compare medians across reps.
#
# For a promotion decision prefer ab.sh, which interleaves the champion and the
# challenger inside one lease hold. This script characterises a single candidate
# (spread, bimodality, a baseline); it does not defend a comparison against
# drift between two separately-measured runs.
#
# Logs land in <task>/runs/<id>.rep<k>_bench.log; collect.py groups them.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
BENCH=${CB_BENCH_CUDA:-/home/meteorix/proj/kernelbench.com/benchmarks/cuda}
VENV="$BENCH/.venv"

# Root before the lease: run.sh re-execs under `sudo -n`, and sudo scrubs the
# environment, which would drop CB_GPU_LEASE and make the inner gpu.sh block on
# a lock this process already holds. See ab.sh for the same note.
[ -r "$VENV/bin/python" ] || exec sudo -n "$0" "$@"

if [ "${1:-}" = "--leased" ]; then
  shift
  nn=$1 reps=$2 task=$3 id=$4
  for k in $(seq 1 "$reps"); do
    log="$task/runs/${id}.rep${k}_bench.log"
    echo "== rep $k/$reps -> $(basename "$log")"
    "$here/run.sh" bench "$nn" 2>&1 | tee "$log" | grep -E '^peak_fraction' || true
  done
  exit 0
fi

nn=${1:?problem number, e.g. 01}
cand=${2:?candidate id, e.g. c2_bm128}
reps=${3:-3}

task=$(echo "$here"/"$nn"-*)
id=${cand%%_*}
mkdir -p "$task/runs"
export CB_TASK="$nn"

# Refuse to clobber. runs/ is gitignored, so these logs are the only copy of the
# evidence a ledger entry was derived from — and the silent failure is worse than
# the loss: re-running 3 reps over a 7-rep set leaves 4 old and 3 new logs that
# collect.py will happily pool into one "7-rep" row measured hours apart under
# different ambient load. That is a fabricated measurement. This cost c8's first
# three reps once; it should not cost anything again.
clash=$(ls "$task/runs/${id}".rep*_bench.log 2>/dev/null || true)
if [ -n "$clash" ] && [ "${CB_FORCE:-0}" != "1" ]; then
  echo "repeat.sh: $id already has rep logs:" >&2
  echo "$clash" | sed 's/^/  /' >&2
  echo "repeat.sh: move them aside (they are the evidence for a ledger row), or" >&2
  echo "repeat.sh: set CB_FORCE=1 to overwrite deliberately." >&2
  exit 2
fi

"$here/run.sh" use "$nn" "$cand"
"$here/run.sh" prebuild "$nn" >/dev/null   # compile off the lease

# One lease for all reps, not one per rep. gpu.sh is reentrant so the inner
# run.sh calls pass straight through. Re-queueing per rep would let another
# worker interleave between this candidate's reps, reintroducing exactly the
# run-to-run variation the reps exist to average out.
exec "$here/gpu.sh" "reps-$nn-$id" -- "$0" --leased "$nn" "$reps" "$task" "$id"

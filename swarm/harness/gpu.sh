#!/usr/bin/env bash
# Single-GPU lease. Every command that touches the GPU goes through here.
#
#   ./gpu.sh <label> -- <command...>
#
# There is one RTX 2070 and up to four workers. It is also the scoring
# instrument, not just the compute device, so the lease exists for two reasons
# and the second is the important one:
#
#   1. Two concurrent CUDA processes on an 8 GB card is an OOM, and two
#      concurrent `ncu` runs simply fail.
#   2. A measurement taken while another worker is also on the GPU is not
#      comparable to one taken alone. KDA-PLAYBOOK.md rule 5: this card drives
#      the display, idles at 315 MHz, cannot have its clocks locked under WSL2,
#      and one task-01 shape is bimodal across a 1.64x clock-bin gap. Ambient
#      load is a variable we can actually remove, so we remove it.
#
# flock is held on fd 9 for the child's whole lifetime and released by the
# kernel on exit *or* crash, so a killed worker — or a killed leader session —
# cannot leak the lease. There is no stale-lock recovery path because there is
# no stale-lock failure mode.
#
# Exit 75 (EX_TEMPFAIL) means "lease busy, try later": callers should go do CPU
# work rather than block. See CB_GPU_WAIT.
set -euo pipefail

# Become root before touching the lock, uniformly. This system runs
# `fs.protected_regular = 2`, which forbids opening for write a file you do not
# own inside a sticky world-writable directory such as /tmp — and it applies to
# root too, so a 0666 mode does not help. run.sh, repeat.sh, ab.sh and
# profile.sh all re-exec under `sudo -n` anyway because the bench venv is
# root-owned, so without this the lease file would belong to whichever user got
# there first and be un-openable by the other, with a bare "Permission denied".
if [ "$(id -u)" -ne 0 ]; then
  exec sudo -n --preserve-env=CB_GPU_LEASE,CB_TASK,CB_GPU_WAIT,CB_GPU_LOCK,CB_GPU_LOG,CB_BENCH,CB_NVSMI,CB_GPU_ALLOW_FOREIGN,CB_SKIP_BENCH_GUARD "$0" "$@"
fi

LOCK=${CB_GPU_LOCK:-/tmp/cuda-bench.gpu.lock}
HOLDER="$LOCK.holder"
LOG=${CB_GPU_LOG:-/tmp/cuda-bench.gpu.log}
WAIT=${CB_GPU_WAIT:-3600}
BENCH=${CB_BENCH:-/home/meteorix/proj/kernelbench.com}
NVSMI=${CB_NVSMI:-/usr/lib/wsl/lib/nvidia-smi}

label=${1:?usage: gpu.sh <label> -- <command...>}
shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "gpu.sh: no command given" >&2; exit 2; }

# Reentrant: if an ancestor already holds the lease, just run. This is what lets
# repeat.sh and ab.sh take the lease *once* and call run.sh N times inside it.
# Without it each rep would re-queue, another worker could interleave between
# reps, and ab.sh's paired comparison — the whole point of which is that both
# arms see the same conditions — would silently stop being paired.
if [ -n "${CB_GPU_LEASE:-}" ]; then
  exec "$@"
fi
export CB_GPU_LEASE=$$

exec 9>"$LOCK"

queued=$SECONDS
if ! flock -w "$WAIT" -x 9; then
  echo "gpu.sh: lease busy after ${WAIT}s; holder: $(cat "$HOLDER" 2>/dev/null || echo unknown)" >&2
  exit 75
fi
waited=$((SECONDS - queued))

# ---- preflight guards, inside the lease ------------------------------------
# Both run after acquiring, because both are about the state the measurement is
# about to be taken in.

# 1. Nothing else resident on the card. 8 GB total, and a leftover CUDA context
#    or a loaded ollama model (~5.5 GB) turns a benchmark into an OOM or, worse,
#    into a quietly slower number.
if [ -x "$NVSMI" ] && [ "${CB_GPU_ALLOW_FOREIGN:-0}" != "1" ]; then
  apps=$("$NVSMI" --query-compute-apps=pid,process_name,used_memory \
         --format=csv,noheader 2>/dev/null || true)
  if [ -n "$apps" ]; then
    echo "gpu.sh: foreign GPU processes resident, refusing to measure:" >&2
    echo "$apps" >&2
    echo "gpu.sh: clear them (e.g. 'ollama stop'), or set CB_GPU_ALLOW_FOREIGN=1" >&2
    exit 1
  fi
fi

# 2. The bench submodule is shared by all four workers and its check.py,
#    benchmark.py, shapes.py, reference.py, problem.yaml and PROMPT.txt are
#    bench-rule-forbidden to edit. An edit by one worker silently invalidates
#    every other worker's numbers, so catch it at lease time rather than at
#    merge time. solution.py and __pycache__ are gitignored by the bench, so a
#    clean tree here means exactly "nobody touched the rules".
if [ "${CB_SKIP_BENCH_GUARD:-0}" != "1" ]; then
  dirty=$(git -C "$BENCH" status --porcelain 2>/dev/null || true)
  if [ -n "$dirty" ]; then
    echo "gpu.sh: bench submodule is dirty — every task's numbers are suspect:" >&2
    echo "$dirty" >&2
    echo "gpu.sh: restore it (git -C $BENCH checkout -- .) before measuring" >&2
    exit 1
  fi
fi

# ---- hold ------------------------------------------------------------------
printf '%s %s %s %s\n' "$$" "$label" "${CB_TASK:-?}" "$(date -Is)" > "$HOLDER"
trap 'rm -f "$HOLDER"' EXIT

started=$SECONDS
set +e
"$@"
status=$?
set -e
held=$((SECONDS - started))

# Non-fatal: under `set -e` a failed append would exit with the *printf's*
# status and silently replace the command's own. The audit log is for the
# leader's starvation check; losing a line must never change what a caller sees.
printf '%s label=%s task=%s waited=%ss held=%ss status=%s load=%s\n' \
  "$(date -Is)" "$label" "${CB_TASK:-?}" "$waited" "$held" "$status" \
  "$(cut -d' ' -f1 /proc/loadavg)" >> "$LOG" 2>/dev/null || true

exit "$status"

#!/usr/bin/env bash
# Leader liveness probe. Answers "is this trial healthy right now", which no other
# artefact does: STATUS.md reports what happened after a cycle, and the stall
# detector only runs at cycle boundaries.
#
#   ./wstat.sh                 # all worktrees
#
# THREE SIGNALS, AND ONLY TWO OF THEM WORK.
#
# * The process table, keyed by worktree cwd — the reliable one.
# * The GPU lease audit log — the other reliable one.
# * File mtimes — DO NOT USE. A worker that has stopped writing and started running
#   looks identical to a dead one. This cost the leader two wrong calls in trial 1:
#   once declaring a live round dead, once nearly missing a genuinely stuck worker.
#
# A long-running call is not necessarily stuck. Three states look alike at a glance
# and this script separates them:
#   BUSY    holding the lease, doing long legitimate work (a full sweep can be 20 min)
#   QUEUED  blocked in flock waiting for the lease — expected, not a problem
#   STUCK?  neither, and past the threshold — worth a look
set -euo pipefail

WT_ROOT=${CB_WT_ROOT:-/home/meteorix/proj-wt}
LOG=${CB_GPU_LOG:-/tmp/cuda-bench.gpu.log}
LOCK=${CB_GPU_LOCK:-/tmp/cuda-bench.gpu.lock}
STUCK_S=${CB_STUCK_SECONDS:-300}

echo "== $(date -Is)  load=$(cut -d' ' -f1-3 /proc/loadavg)"

holder=$(sudo -n cat "$LOCK.holder" 2>/dev/null || true)
if [ -n "$holder" ]; then
  set -- $holder
  echo "   GPU lease: HELD by $2 (task $3) since $4"
else
  echo "   GPU lease: free"
fi

for wt in "$WT_ROOT"/*/; do
  [ -d "$wt" ] || continue
  tag=$(basename "$wt")
  n=0; oldest=0; ocmd=idx; queued=0
  for p in $(pgrep -f . 2>/dev/null); do
    c=$(sudo -n readlink /proc/$p/cwd 2>/dev/null) || continue
    case "$c" in *"$tag"*) ;; *) continue ;; esac
    n=$((n+1))
    e=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' '); [ -n "${e:-}" ] || continue
    cmd=$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ')
    [ "$cmd" = flock ] && queued=1
    if [ "$e" -gt "$oldest" ]; then oldest=$e; ocmd=$cmd; fi
  done
  br=$(git -C "$wt" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')
  ahead=$(git -C "$wt" rev-list --count "master..$br" 2>/dev/null || echo 0)

  state=idle
  if [ "$n" -gt 0 ]; then
    if [ "$queued" = 1 ]; then state=QUEUED
    elif [ -n "$holder" ] && echo "$holder" | grep -q " ${tag#cb} "; then state=BUSY
    elif [ "$oldest" -gt "$STUCK_S" ]; then state="STUCK?"
    else state=working; fi
  fi
  printf "   %-6s %-8s procs=%-2s longest=%-6s %-8s commits=%s\n" \
    "$tag" "$state" "$n" "${oldest}s" "$ocmd" "$ahead"
done

echo "   -- lease log (last 5) --"
sudo -n tail -5 "$LOG" 2>/dev/null | sed 's/^/   /' || echo "   (none yet)"

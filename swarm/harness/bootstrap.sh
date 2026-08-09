#!/usr/bin/env bash
# Make a trial runnable. Idempotent — safe to re-run.
#
#   ./swarm/harness/bootstrap.sh            # from the trial root
#
# `start-trial.sh` calls this, and so must anyone who CLONES a trial, because two
# of the three things a trial needs are not in git:
#
#   1. The submodule contents (`kernelbench.com`, `kernel-design-agents`). A clone
#      records the pointers, not the trees.
#   2. The `.venv` symlink. The bench is 58 MB of git and 4.7 GB of installed
#      torch; the venv is shared, so it cannot be committed.
#   3. The exclude that stops that symlink showing as untracked content INSIDE the
#      bench submodule. This is the subtle one: excludes live in
#      `.git/modules/<path>/info/exclude`, which is never cloned. Without it the
#      bench reads dirty, and `gpu.sh` aborts every run on a dirty bench --
#      correctly, since a modified instrument invalidates every number in the
#      trial. The failure looks like a permissions problem and is not.
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"
[ -f TRIAL.md ] || { echo "bootstrap.sh: run me from a trial (no TRIAL.md here)" >&2; exit 2; }

venv=${CB_SHARED_VENV:-/home/meteorix/proj/kernelbench.com/benchmarks/cuda/.venv}

echo "== submodules"
git -c protocol.file.allow=always submodule update --init

bench=kernelbench.com/benchmarks/cuda
if [ -d "$bench" ]; then
  if [ -e "$bench/.venv" ]; then
    echo "== venv already linked"
  elif [ -d "$venv" ]; then
    ln -s "$venv" "$bench/.venv"
    echo "== venv -> $venv"
  else
    echo "== venv NOT FOUND at $venv" >&2
    echo "   Set CB_SHARED_VENV, or create one with:" >&2
    echo "     cd $bench && uv sync && ./scripts/patch_torch.sh" >&2
  fi

  ex="$(git -C kernelbench.com rev-parse --git-dir)/info/exclude"
  if ! grep -qx "benchmarks/cuda/.venv" "$ex" 2>/dev/null; then
    mkdir -p "$(dirname "$ex")"
    echo "benchmarks/cuda/.venv" >> "$ex"
    echo "== excluded the venv symlink inside the bench submodule"
  fi

  dirty=$(git -C kernelbench.com status --porcelain || true)
  if [ -n "$dirty" ]; then
    echo "== WARNING: bench is dirty, gpu.sh will refuse to run:" >&2
    echo "$dirty" | sed 's/^/     /' >&2
  else
    echo "== bench clean"
  fi
fi

cat <<EOF

ready. next:
  source env.sh
  ./swarm/harness/gpu.sh smoke -- true
EOF

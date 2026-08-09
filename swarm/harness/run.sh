#!/usr/bin/env bash
# Thin wrapper: run the bench's own check.py / benchmark.py for one problem with
# the toolchain this machine needs on PATH (ninja from the bench venv, nvcc from
# /usr/local/cuda — the playbook notes nvcc is not on PATH here).
#
#   ./run.sh use 01 c0_simt    install a candidate as the problem's solution.py
#   ./run.sh prebuild 01       compile the installed solution without the GPU
#   ./run.sh check 01          bench's own validator; must print PASS
#   ./run.sh bench 01          bench's own 6-shape sweep; prints the geomean
#   ./run.sh py 01 -c '...'    scratch python inside the bench venv
#
# The venv interpreter is a symlink into root-owned /root/.local/share/uv, so
# every command re-execs under `sudo -n` when it is not readable as the calling
# user. A side effect worth knowing: HOME becomes /root, so load_inline builds
# land in /root/.cache/torch_extensions, on ext4.
#
# WORK is derived from this script's own directory, not hardcoded. Each worker
# runs from its own git worktree, and a hardcoded WORK would send every worker's
# `use` to the *leader's* candidates — i.e. every worker would silently bench
# somebody else's code. BENCH stays absolute on purpose: one venv, one set of
# problem definitions, one solution.py slot per problem, shared by all workers.
# The four problems are disjoint directories, so those slots never collide.
#
# check and bench acquire the GPU lease (gpu.sh). `use`, `prebuild` and `py` do
# not: `use` is a copy, `prebuild` is deliberately GPU-free, and `py` is a
# scratch hatch whose caller decides. Wrap `py` yourself if it launches kernels.
set -euo pipefail

BENCH=${CB_BENCH_CUDA:-/home/meteorix/proj/kernelbench.com/benchmarks/cuda}
WORK=$(cd "$(dirname "$0")" && pwd)
VENV="$BENCH/.venv"

# --preserve-env: sudo scrubs the environment, and dropping CB_GPU_LEASE here
# would make gpu.sh below block on a lock an ancestor already holds. Callers
# that take the lease themselves (ab.sh, repeat.sh) become root first for the
# same reason; this is the belt to their braces.
[ -r "$VENV/bin/python" ] || \
  exec sudo -n --preserve-env=CB_GPU_LEASE,CB_TASK,CB_BENCH_CUDA "$0" "$@"

export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=7.5
# Four concurrent workers on 12 cores. Uncapped ninja would let one worker's
# build starve whoever currently holds the lease, which is the one process whose
# timing we care about.
export MAX_JOBS=${MAX_JOBS:-2}

cmd=${1:?use|prebuild|check|bench|py}; nn=${2:?problem number, e.g. 01}; shift 2 || true
dir=$(echo "$BENCH"/problems-rtx2070/"$nn"_*)
task=$(echo "$WORK"/"$nn"-*)

export CB_TASK="$nn"

case "$cmd" in
  use)
    src="$task/candidates/${1:?candidate id, e.g. c0_simt}.py"
    [ -f "$src" ] || { echo "no such candidate: $src" >&2; exit 2; }
    cp "$src" "$dir/solution.py"
    rm -rf "$dir/__pycache__"
    echo "installed $(basename "$src") -> $dir/solution.py"
    ;;

  prebuild)
    # Populate /root/.cache/torch_extensions without holding the lease. The
    # candidates call load_inline at module scope, so importing builds. With no
    # visible device and an explicit arch list, nvcc has nothing to query and no
    # CUDA context is created — so this is safe to run while another worker is
    # measuring, and the leased run afterwards starts already compiled.
    cd "$dir"
    exec env CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -c \
      "import sys; sys.path.insert(0, '.'); import solution; print('prebuilt', solution.__name__)"
    ;;

  check) cd "$dir"; exec "$WORK/gpu.sh" "check-$nn" -- "$VENV/bin/python" check.py "$@" ;;
  bench) cd "$dir"; exec "$WORK/gpu.sh" "bench-$nn" -- "$VENV/bin/python" benchmark.py "$@" ;;
  py)    cd "$dir"; exec "$VENV/bin/python" "$@" ;;
  *) echo "unknown: $cmd" >&2; exit 2 ;;
esac

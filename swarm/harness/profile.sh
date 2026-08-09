#!/usr/bin/env bash
# Nsight Compute one kernel launch of the installed solution.py.
#
#   ./profile.sh 01 gemv1_swiglu 1        # kernel regex, T
#   ./profile.sh 01 gemm1_swiglu 4096
#
# Reports land in <task>/profile/<kernel>_T<T>/, following the layout
# ncu-report-skill expects (one directory per run, REPORT.md written by hand
# alongside the .ncu-rep).
#
# Three things this wrapper exists to get right, all from KDA-PLAYBOOK.md:
#   * -k "regex:<kernel>" — without it --launch-count 1 profiles whatever
#     launched first, usually a warmup or a library kernel.
#   * --profile-from-start off, paired with prof_driver.py's cudaProfilerStart,
#     so the captured launch is a warm one.
#   * --set full replays ~34 passes, which is why this profiles a single
#     launch and not a sweep.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
nn=${1:?problem number, e.g. 01}
kernel=${2:?kernel name regex, e.g. gemv1_swiglu}
tt=${3:-1}

BENCH=/home/meteorix/proj/kernelbench.com/benchmarks/cuda
VENV="$BENCH/.venv"
[ -r "$VENV/bin/python" ] || exec sudo -n "$0" "$@"

task=$(echo "$here"/"$nn"-*)
dir=$(echo "$BENCH"/problems-rtx2070/"$nn"_*)

# Namespace the run by whichever candidate is installed. Without this a second
# candidate's profile silently overwrites the first's, which is exactly what
# happened to c6's gemm1 report when c7 was profiled.
# Match any load_inline name=, not `moe_*`. The original pattern hardcoded task
# 01's naming convention, so every task-02/03/04 profile landed in
# profile/unknown_<kernel>_T<n>/ and had to be renamed by hand — and two profiles
# of different candidates would then have collided under `unknown`, which is the
# exact overwrite this namespacing exists to prevent. Same extraction ab.sh uses.
cand=$(sed -n 's/.*name="\([a-z0-9_]*\)".*/\1/p' "$dir/solution.py" | head -1)
out="$task/profile/${cand:-unknown}_${kernel}_T${tt}"
# Announce what is actually installed. This is derived from solution.py, not from
# anything the caller passed, and an A/B leaves whichever arm ran last in place --
# so "I profiled the challenger" is an assumption until this line confirms it.
echo "profiling INSTALLED solution: extension name '${cand:-unknown}' -> $(basename "$out")"
[ -d "$out" ] && echo "NOTE: $out exists and will be overwritten."
mkdir -p "$out"

export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=7.5

export CB_TASK="$nn"

# The longest hold of any leased operation: --set full replays ~34 passes. The
# lease log is what makes that visible to the leader, so a profile starving the
# other workers shows up in STATUS.md instead of being guessed at.
cd "$dir"
"$here/gpu.sh" "ncu-$nn-$kernel" -- \
/usr/local/cuda/bin/ncu \
  --profile-from-start off \
  -k "regex:$kernel" \
  --launch-count 1 \
  --set full \
  --force-overwrite \
  -o "$out/report" \
  "$VENV/bin/python" "$task/tools/prof_driver.py" "$tt"

echo "-> $out/report.ncu-rep"

# source this before using swarm/harness — it points the harness at THIS trial's
# pinned bench rather than the rig's checkout.
_self=${BASH_SOURCE[0]:-$0}
_here=$(cd "$(dirname "$_self")" && pwd)
export CB_BENCH_CUDA="$_here/kernelbench.com/benchmarks/cuda"
export CB_WT_ROOT="${CB_WT_ROOT:-$_here-wt}"
unset _self _here

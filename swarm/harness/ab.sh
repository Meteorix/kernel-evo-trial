#!/usr/bin/env bash
# Paired A/B: champion and challenger, interleaved, inside ONE lease hold.
#
#   ./ab.sh 01 c8_stage c10_mma 4
#
# Why this exists, and why it replaces "run repeat.sh on the challenger and
# compare against the champion's old number":
#
# The champion's recorded geomean was measured at some earlier time, at whatever
# clocks the card happened to be at, with whatever else was running. This card
# drives the display, idles at 315 MHz, boosts to 2160, and cannot have its
# clocks locked under WSL2. KDA-PLAYBOOK.md rule 5 already cost this deck one
# wrong promotion that way (c1 over c2, on a T=1 outlier). With four workers
# sharing the machine there is now a *second* uncontrolled variable — how many
# of them were compiling at the time.
#
# Interleaving champ, chal, champ, chal... inside a single lease makes both arms
# sample the same clock excursions and the same ambient load, so the difference
# between them is attributable to the code. It does not make the absolute
# numbers portable across sessions — nothing can — which is why ambient.txt is
# recorded alongside.
#
# Read the result with ms_min as well as the median. rule 5's second half: one
# task-01 shape is bimodal across a 1.64x clock-bin gap, and a median over reps
# reports which bin came up more often, not which kernel is faster.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
BENCH=${CB_BENCH_CUDA:-/home/meteorix/proj/kernelbench.com/benchmarks/cuda}
VENV="$BENCH/.venv"

# Become root before taking the lease, not after. run.sh re-execs under `sudo -n`
# when the venv is unreadable, and sudo scrubs the environment — so a lease taken
# as the calling user would lose CB_GPU_LEASE on the way into run.sh, and the
# inner gpu.sh would block on a lock this very process already holds.
[ -r "$VENV/bin/python" ] || exec sudo -n "$0" "$@"

# --- leased body ------------------------------------------------------------
if [ "${1:-}" = "--leased" ]; then
  shift
  nn=$1 champ=$2 chal=$3 reps=$4 out=$5
  ida=${champ%%_*} idb=${chal%%_*}

  # Warm the clocks before the first measured rep, and throw the result away.
  # Measured on task 02: arm A's rep 1, first three shapes only, ran ~32% slow in
  # 3 of 4 runs — the boost ramp inside the first ~250 ms of the lease decaying
  # through the sweep. It appeared only when the GPU was idle before the lease and
  # not when the run queued 163 s behind another worker. Since arm A is always the
  # champion, the bias ran systematically AGAINST the champion, i.e. toward
  # promoting: exactly the direction a promotion harness must not be biased.
  # KDA-PLAYBOOK rule 5 says warm the clocks before measuring; the lease made it
  # easy to forget that the *lease itself* can start cold.
  echo "== warming clocks (discarded)"
  "$VENV/bin/python" -c "
import torch
a = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
for _ in range(400): a = a @ a * 0.0 + a
torch.cuda.synchronize()
" >/dev/null 2>&1 || true

  for k in $(seq 1 "$reps"); do
    # Alternate which arm runs first. Even with the warm-up, any residual
    # first-position effect then cancels across reps instead of accumulating on
    # one arm. Costs nothing and does not disturb the interleaving.
    if [ $((k % 2)) -eq 1 ]; then order="$champ:$ida $chal:$idb"; else order="$chal:$idb $champ:$ida"; fi
    for arm in $order; do
      cand=${arm%%:*} cid=${arm##*:}
      log="$out/${cid}.ab${k}_bench.log"
      echo "== rep $k/$reps  $cid"
      "$here/run.sh" use "$nn" "$cand" >/dev/null
      "$here/run.sh" bench "$nn" 2>&1 | tee "$log" | grep -E '^peak_fraction' || true
      printf '%s %s rep%s load=%s\n' "$(date -Is)" "$cid" "$k" \
        "$(cut -d' ' -f1 /proc/loadavg)" >> "$out/ambient.txt"
    done
  done
  exit 0
fi

# --- setup ------------------------------------------------------------------
nn=${1:?problem number, e.g. 01}
champ=${2:?champion candidate id, e.g. c8_stage}
chal=${3:?challenger candidate id, e.g. c10_mma}
reps=${4:-4}

task=$(echo "$here"/"$nn"-*)
ida=${champ%%_*} idb=${chal%%_*}
# Never reuse a run directory. Replication is required, not optional: a single
# paired A/B measures one draw of the error, and task 01 measured c11's T=4096 at
# +0.71% in one run and -0.11% in the next -- a 0.82-point swing against a 0.23%
# within-session floor. So a second run of the same pair is the normal case, and
# keying the directory on the pair alone silently destroyed the first run's logs.
# Same bug class as repeat.sh's, in the file written beside it; auto-numbering
# rather than refusing, because here the repeat is the point.
out="$task/runs/ab_${ida}_vs_${idb}"
if [ -d "$out" ]; then
  n=2
  while [ -d "${out}_run${n}" ]; do n=$((n + 1)); done
  out="${out}_run${n}"
  echo "ab.sh: prior run exists; this is run $n -> $(basename "$out")"
fi
mkdir -p "$out"
: > "$out/ambient.txt"

export CB_TASK="$nn"

# load_inline keys its build directory by the extension `name=`. Two candidates
# sharing a name would thrash one build dir and recompile on every swap — inside
# the lease, where it is most expensive. Distinct names are the deck's existing
# convention (profile.sh relies on it too); warn rather than fail, since a
# rebuild is slow, not wrong.
nm() { sed -n 's/.*name="\([a-z0-9_]*\)".*/\1/p' "$task/candidates/$1.py" | head -1; }
if [ "$(nm "$champ")" = "$(nm "$chal")" ]; then
  echo "ab.sh: WARNING both candidates use extension name '$(nm "$champ")';" >&2
  echo "ab.sh: every swap will recompile under the lease. Rename one." >&2
fi

# Compile both off the lease so the leased section is measurement only.
for c in "$champ" "$chal"; do
  "$here/run.sh" use "$nn" "$c" >/dev/null
  "$here/run.sh" prebuild "$nn" >/dev/null
done

"$here/gpu.sh" "ab-$nn-$ida-$idb" -- "$0" --leased "$nn" "$champ" "$chal" "$reps" "$out"

# --- report -----------------------------------------------------------------
echo
"$VENV/bin/python" "$here/collect.py" "$out"/*_bench.log | tee "$out/compare.csv"
echo
echo "-> $out/  (compare.csv, ambient.txt, ${reps} paired reps)"
# Say which candidate is left installed. profile.sh keys its output directory off
# the INSTALLED solution.py, so a profile run straight after an A/B silently
# profiles whichever arm finished last -- which cost task 04 an archived report
# when it re-profiled the champion believing it had the challenger.
echo "NOTE: solution.py currently holds ${chal} (whichever arm ran last)."
echo "      Run '../run.sh use $nn <candidate>' before ../profile.sh, or you will profile this one."
echo "Promote on ms_min as well as the median — see KDA-PLAYBOOK.md rule 5."

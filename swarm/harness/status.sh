#!/usr/bin/env bash
# Append one iteration to a task's STATUS.md. The file is APPEND-ONLY.
#
#   ./status.sh 01 c1_plan promoted --score 0.0741 <<'EOF'
#   **Hypothesis** — the 4 host syncs in the routing plan cost 61.5% of the T=1
#   forward; removing them is worth ~1.19x on the geomean and nothing elsewhere.
#   **Result** — PROMOTED. T=1 2.66 -> 1.31 ms; the five prefill shapes moved
#   -0.3% to +0.4%, inside the 1% null band, as predicted.
#   **Notes** — the null arm (c1n) was flat, so the win is attributed to the syncs
#   and not to the nrows<=0 guard that came with them.
#   EOF
#
# WHY APPEND-ONLY. A status file that is rewritten each cycle answers "where is
# this task now" and destroys "how did it get here". The second question is the
# one a reviewer actually has: which hypothesis was posed, what came back, and
# what was believed at the time. Overwriting also lets a stale summary survive a
# killed cycle looking authoritative -- exactly what happened to task 04 in t2,
# whose header still claimed cycle 1 after it had committed a cycle-2 sweep.
#
# So the champion is NOT a field here. It is whatever the last `promoted` entry
# says, which cannot drift from the log because it IS the log.
#
# The commit id recorded is HEAD at append time -- i.e. the commit containing the
# work being described. Commit the work FIRST, then append, then commit the
# STATUS change. Two commits, and the entry points at the right one.
set -euo pipefail

usage() {
  echo "usage: status.sh <NN> <id> <verdict> [--score S] [--replicated] < body" >&2
  echo "  verdict: promoted | rejected | baseline | note | selftest | blocked" >&2
  exit 2
}

[ $# -ge 3 ] || usage
nn=$1; id=$2; verdict=$3; shift 3
score=""; repl=""

while [ $# -gt 0 ]; do
  case "$1" in
    --score) score=${2:?--score needs a value}; shift 2 ;;
    --replicated) repl=" · replicated"; shift ;;
    *) usage ;;
  esac
done

case "$verdict" in
  promoted|rejected|baseline|note|selftest|blocked) ;;
  *) echo "status.sh: unknown verdict '$verdict'" >&2; usage ;;
esac

here=$(cd "$(dirname "$0")" && pwd)
dir=$(echo "$here"/"$nn"-*)
[ -d "$dir" ] || { echo "status.sh: no task dir for $nn under $here" >&2; exit 2; }
f="$dir/STATUS.md"
[ -f "$f" ] || { echo "status.sh: $f does not exist" >&2; exit 2; }

body=$(cat)
[ -n "${body// }" ] || { echo "status.sh: empty body on stdin; an entry with no content is not an entry" >&2; exit 2; }

for want in Hypothesis Result Notes; do
  echo "$body" | grep -q "\*\*$want\*\*" || {
    echo "status.sh: body is missing a **$want** line. All three are required —" >&2
    echo "  an entry without a hypothesis cannot be reviewed, and one without a" >&2
    echo "  result is a plan, not a record." >&2
    exit 2
  }
done

sha=$(git -C "$dir" rev-parse --short HEAD 2>/dev/null || echo "no-git")
ts=$(date -Is)
[ -n "$score" ] && score=" · geomean \`$score\`"

{
  printf '\n### %s · `%s`%s%s\n\n' "$id" "$verdict" "$score" "$repl"
  printf '%s · commit `%s`\n\n' "$ts" "$sha"
  printf '%s\n' "$body"
} >> "$f"

echo "appended to $f:"
printf '  %s · %s%s%s · %s\n' "$id" "$verdict" "${score//\`/}" "$repl" "$sha"

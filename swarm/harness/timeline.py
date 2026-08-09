#!/usr/bin/env python3
"""Draw the worker trace for an evolution trial as a self-contained HTML page.

    ./timeline.py                      # trial dir = cwd, output ./timeline.html
    ./timeline.py --trial ../trials/2026-08-08-t2 --out /tmp/t.html
    ./timeline.py --open               # also print the path to hand to Artifact

Three signals, joined on one time axis:

  * the GPU lease audit log  -> who held the card, for how long, and who queued
  * per-branch `git log`     -> commits, attributed to a task by the paths touched
  * merge commits on master  -> cycle boundaries per task

WHY THIS EXISTS. STATUS.md says what a worker concluded; wstat.sh says whether it
is alive right now. Neither shows the shape of a trial over time -- where the card
actually went, which single lease caused the queueing, how far apart the cycles
drifted. Those only appear when the three logs are laid on a shared axis, and every
one of them is already on disk. This just joins them.

READ THE OCCUPANCY NUMBER CAREFULLY. A lease covers process start, compile, and
teardown as well as kernel execution, so held-time is an UPPER bound on GPU busy
time -- generally a loose one. Use it to answer "is the card the bottleneck", which
it answers safely in the direction that matters, and not as a utilization figure.

ONE KNOWN OVERCOUNT. Commits are deduped by SHA, so a worker that rebased onto a
master which already contains its merged work shows those commits TWICE -- once as
the merged original, once as the replayed copy. The lane is still shaped right; the
count is high. Deduping by patch-id instead would fix it and costs a `git patch-id`
per commit, which was not worth it at this scale.
"""
import argparse, json, os, re, subprocess, sys, time
from collections import defaultdict

HUES = [("--t1", "#B96F22", "#E3A055"), ("--t2", "#1F8479", "#54BCB0"),
        ("--t3", "#6154B4", "#9A90E2"), ("--t4", "#AE4463", "#DB7B94"),
        ("--t5", "#3C7CA8", "#6FAFD8"), ("--t6", "#8A6A2F", "#C6A05C")]


def sh(*a, **kw):
    return subprocess.run(a, capture_output=True, text=True, **kw).stdout


def git(repo, *a):
    return sh("git", "-C", repo, *a)


def parse_leases(path, t0, tasks):
    """Lease lines are written at RELEASE, so a record's span is [ts-waited-held, ts].

    The `task=` field is often `?` -- gpu.sh only fills it when CB_TASK survives the
    sudo hop -- so fall back to the task number embedded in the label (`check-03`,
    `ab-02-c0-c1`, `ncu-01-moe_gemm1`). Between the two, attribution is complete.
    """
    out = []
    try:
        raw = sh("sudo", "-n", "cat", path)
    except Exception:
        raw = ""
    if not raw:
        try:
            raw = open(path).read()
        except OSError:
            print(f"timeline: cannot read lease log {path}", file=sys.stderr)
            return out
    for ln in raw.splitlines():
        m = re.match(r"(\S+) label=(\S+) task=(\S+) waited=(\d+)s held=(\d+)s status=(\d+)", ln)
        if not m:
            continue
        ts, label, task, waited, held, st = m.groups()
        end = iso(ts)
        if end < t0:
            continue
        if task not in tasks:
            g = re.search(r"(?:^|[-_])(\d{2})(?:$|[-_.])", label)
            task = g.group(1) if g and g.group(1) in tasks else "??"
        if task == "??":
            continue
        out.append([task, int(end - int(waited) - int(held) - t0), int(waited), int(held), label, int(st)])
    return sorted(out, key=lambda r: r[1])


def iso(s):
    return int(time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")))


def collect_commits(repo, prefix, tasks, t0, branches):
    """One commit can be reachable from several branches; dedupe by SHA, then
    attribute by the task dir it touches. A commit touching none (a leader edit to
    the playbook, a merge) is not a worker commit and is dropped from the lanes.

    `prefix` is "" when the trial IS the repo (the normal case since trials became
    their own repos) and "<dir>/" when it is a subdirectory of a larger one. Match
    on the START of each path, not anywhere in the blob: with an empty prefix a
    substring test for "01-" hits every path that merely contains it."""
    seen, commits, merges = {}, defaultdict(list), []
    for b in branches:
        for ln in git(repo, "log", "--format=%H %at %P", f"--since=@{t0}", b).splitlines():
            p = ln.split()
            if len(p) < 2 or p[0] in seen:
                continue
            seen[p[0]] = (int(p[1]), len(p) - 2)
    for sha, (at, nparents) in seen.items():
        paths = git(repo, "show", "--format=", "--name-only", sha).splitlines()
        hit = sorted({t for t in tasks
                      if any(p.startswith(f"{prefix}{t}-") for p in paths)})
        subj = git(repo, "log", "-1", "--format=%s", sha).strip()
        if nparents > 1:
            g = re.search(r"/(\d{2})-", subj) or re.search(r"\b(\d{2})\b", subj)
            merges.append([at - t0, subj[:60], hit[0] if len(hit) == 1 else (g.group(1) if g else None)])
        elif len(hit) == 1:
            commits[hit[0]].append(at - t0)
    return {k: sorted(v) for k, v in commits.items()}, sorted(merges)


def build(trial, repo, log_path, out):
    trial = os.path.abspath(trial)
    rel = os.path.relpath(trial, repo)
    prefix = "" if rel == "." else rel + "/"
    name = os.path.basename(trial)

    tasks = sorted(d.split("-")[0] for d in os.listdir(trial)
                   if re.match(r"\d\d-", d) and os.path.isdir(os.path.join(trial, d)))
    labels = {d.split("-")[0]: d for d in os.listdir(trial) if re.match(r"\d\d-", d)}
    if not tasks:
        sys.exit(f"timeline: no NN-* task dirs under {trial}")

    born = git(repo, "log", "--diff-filter=A", "--format=%at", "--", f"{prefix}TRIAL.md").split()
    t0 = int(born[-1]) if born else int(os.path.getmtime(trial))
    t1 = int(time.time())
    span = max(t1 - t0, 60)

    branches = ["master"] + [b.strip() for b in git(repo, "branch", "--format=%(refname:short)").splitlines()
                             if re.search(r"/\d\d-", b)]
    leases = parse_leases(log_path, t0, set(tasks))
    commits, merges = collect_commits(repo, prefix, set(tasks), t0, branches)

    lanes = []
    for i, t in enumerate(tasks):
        bounds = [m[0] for m in merges if m[2] == t] + [span]
        cyc, prev = [], 0
        for b in bounds:
            cyc.append([prev, b])
            prev = b
        lanes.append(dict(id=t, name=labels[t].replace("-", " ", 1), var=HUES[i % len(HUES)][0], cycles=cyc))

    held = sum(r[3] for r in leases)
    wait = sum(r[2] for r in leases)
    worst = max(leases, key=lambda r: r[3]) if leases else None
    hh = sorted(r[3] for r in leases)
    med = hh[len(hh) // 2] if hh else 0

    data = dict(name=name, t0=t0, span=span, lanes=lanes, leases=leases,
                commits=commits, merges=[[m[0], m[1]] for m in merges],
                held=held, wait=wait, ncommit=sum(len(v) for v in commits.values()),
                median=med, occ=round(held / span * 100, 1),
                worst=[worst[4], worst[3], worst[0]] if worst else None,
                started=time.strftime("%H:%M:%S", time.localtime(t0)),
                nowstr=time.strftime("%H:%M:%S", time.localtime(t1)))

    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(out, "w") as f:
        f.write(html)
    return data, out


TEMPLATE = r"""<title>Worker trace</title>
<style>
:root{--page:#EDF1F4;--surface:#FCFDFE;--sunk:#E4EAEF;--ink:#0E141A;--ink-2:#3B4956;
--muted:#6B7885;--faint:#93A0AC;--rule:#D3DBE3;--rule-2:#E6ECF1;--grid:#DCE3EA;
--t1:#B96F22;--t2:#1F8479;--t3:#6154B4;--t4:#AE4463;--t5:#3C7CA8;--t6:#8A6A2F}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--page:#0E1216;--surface:#161B21;--sunk:#11161B;--ink:#E6EBF0;--ink-2:#B4C0CB;
--muted:#8593A0;--faint:#5D6975;--rule:#252D35;--rule-2:#1C232A;--grid:#20272E;
--t1:#E3A055;--t2:#54BCB0;--t3:#9A90E2;--t4:#DB7B94;--t5:#6FAFD8;--t6:#C6A05C}}
:root[data-theme="dark"]{
--page:#0E1216;--surface:#161B21;--sunk:#11161B;--ink:#E6EBF0;--ink-2:#B4C0CB;
--muted:#8593A0;--faint:#5D6975;--rule:#252D35;--rule-2:#1C232A;--grid:#20272E;
--t1:#E3A055;--t2:#54BCB0;--t3:#9A90E2;--t4:#DB7B94;--t5:#6FAFD8;--t6:#C6A05C}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font-size:15px;line-height:1.55;
font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
.wrap{max-width:1180px;margin:0 auto;padding:40px 28px 72px;display:flex;flex-direction:column;gap:34px}
header{display:flex;flex-direction:column;gap:10px;border-bottom:1px solid var(--rule);padding-bottom:22px}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.14em;
text-transform:uppercase;color:var(--muted)}
h1{margin:0;font-size:clamp(26px,3.6vw,38px);line-height:1.1;letter-spacing:-.022em;font-weight:640;text-wrap:balance}
.dek{margin:0;max-width:64ch;color:var(--ink-2);font-size:16px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:1px;
background:var(--rule);border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.stat{background:var(--surface);padding:13px 15px;display:flex;flex-direction:column;gap:3px}
.stat .k{font-size:10.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);
font-family:ui-monospace,Menlo,monospace}
.stat .v{font-size:22px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .n{font-size:12px;color:var(--faint)}
.stat.lead .v{color:var(--t1)}
.panel{border:1px solid var(--rule);border-radius:3px;background:var(--surface);overflow-x:auto}
.trace{min-width:780px}
.axis{position:relative;height:30px;border-bottom:1px solid var(--rule);margin-left:132px}
.axis .tk{position:absolute;top:0;height:100%;border-left:1px solid var(--grid)}
.axis .tk span{position:absolute;left:5px;top:7px;font-size:10.5px;color:var(--muted);
font-family:ui-monospace,Menlo,monospace;letter-spacing:.04em;white-space:nowrap}
.lane{display:flex;align-items:stretch;border-bottom:1px solid var(--rule-2)}
.lane:last-child{border-bottom:0}
.rail{width:132px;flex:0 0 132px;padding:12px 13px;border-right:1px solid var(--rule);
display:flex;flex-direction:column;gap:2px;background:var(--sunk)}
.rail .nm{font-size:12.5px;font-weight:600;display:flex;align-items:center;gap:7px;letter-spacing:-.01em}
.chip{width:9px;height:9px;border-radius:2px;flex:0 0 9px}
.rail .sub{font-size:10.5px;color:var(--muted);font-family:ui-monospace,Menlo,monospace;
font-variant-numeric:tabular-nums}
.track{position:relative;flex:1;height:74px}
.gl{position:absolute;top:0;bottom:0;width:1px;background:var(--grid)}
.cyc{position:absolute;top:0;height:100%;border-left:1px dashed var(--rule)}
.cyc:first-child{border-left:0}
.cyc.alt{background:color-mix(in srgb,var(--sunk) 55%,transparent)}
.cyc .cl{position:absolute;left:6px;top:5px;font-size:10px;color:var(--faint);
font-family:ui-monospace,Menlo,monospace;letter-spacing:.08em}
.blk{position:absolute;top:24px;height:19px;min-width:3px;border-radius:1.5px}
.blk.wait{opacity:.34}
.blk.fail{box-shadow:inset 0 0 0 1.5px var(--page)}
.cm{position:absolute;top:50px;width:2px;height:9px;border-radius:1px;opacity:.85}
.mrg{position:absolute;top:0;bottom:0;width:1px;background:var(--ink-2);opacity:.5}
.legend{display:flex;flex-wrap:wrap;gap:9px 22px;align-items:center;font-size:12px;color:var(--ink-2)}
.lg{display:flex;align-items:center;gap:7px}
.sw{width:22px;height:11px;border-radius:1.5px;background:var(--muted)}
.sw.w{opacity:.34}.sw.c{width:2px;height:11px}
.sw.m{width:1px;height:14px;background:var(--ink-2);opacity:.5}
.tw{border:1px solid var(--rule);border-radius:3px;background:var(--surface);overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--rule-2)}
th{font-size:10.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);
font-weight:500;font-family:ui-monospace,Menlo,monospace}
td.n{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
.note{display:flex;flex-direction:column;gap:7px}
.note h3{margin:0;font-size:13px;letter-spacing:.03em;font-weight:640}
.note p{margin:0;font-size:13.5px;color:var(--ink-2);line-height:1.6}
.note .tag{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;letter-spacing:.12em;
text-transform:uppercase;color:var(--muted)}
b.hl{font-weight:640;color:var(--ink)}
footer{border-top:1px solid var(--rule);padding-top:16px;font-size:12px;color:var(--faint)}
@media (prefers-reduced-motion:no-preference){.lane{animation:in .5s ease both}
@keyframes in{from{opacity:0}to{opacity:1}}}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow" id="eyebrow"></div>
    <h1 id="h1"></h1>
    <p class="dek" id="dek"></p>
  </header>
  <div class="stats" id="stats"></div>
  <div class="panel"><div class="trace">
    <div class="axis" id="axis"></div><div id="lanes"></div>
  </div></div>
  <div class="legend">
    <div class="lg"><span class="sw"></span> holding the card</div>
    <div class="lg"><span class="sw w"></span> queued behind another worker</div>
    <div class="lg"><span class="sw c"></span> commit</div>
    <div class="lg"><span class="sw m"></span> leader merge to master</div>
    <div class="lg"><span class="mono" style="font-size:11px">c1 c2 c3</span> &nbsp;cycle, bounded by its merge</div>
  </div>
  <div class="note" id="reading"></div>
  <div class="tw"><table>
    <thead><tr><th>Task</th><th class="n">Leases</th><th class="n">Held</th>
    <th class="n">Queued</th><th class="n">Commits</th><th>Cycle</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table></div>
  <footer>Generated by <span class="mono">harness/timeline.py</span> from the GPU lease audit log
  and per-branch <span class="mono">git log</span>. Leases never overlap — <span class="mono">flock</span>
  serializes them — so occupancy is a direct sum. Held time includes process start, compile and
  teardown, so it is an <b>upper bound</b> on GPU busy time.</footer>
</div>
<script>
const D = /*__DATA__*/null;
const pct = s => s / D.span * 100;
const hhmm = s => new Date((D.t0 + s) * 1000).toTimeString().slice(0, 5);
const fmt = s => s >= 60 ? Math.floor(s / 60) + "m " + (s % 60) + "s" : s + " s";

document.title = D.name + " — worker trace";
document.getElementById("eyebrow").textContent =
  "Evolution trial " + D.name + " · one GPU, " + D.lanes.length + " agents";
document.getElementById("h1").textContent =
  Math.round(D.span / 60) + " minutes of " + D.lanes.length + " workers sharing one GPU";
document.getElementById("dek").textContent =
  "A lease-log trace of the trial. Every block is a serialized hold on the card; every tick is a " +
  "commit. Window runs " + D.started + " to " + D.nowstr + " local.";

const stats = [
  ["GPU occupancy", D.occ + "%", D.held.toLocaleString() + " s held of " + D.span.toLocaleString() + " s wall", 1],
  ["Leases", D.leases.length, "median hold " + D.median + " s", 0],
  ["Commits", D.ncommit, "across " + D.lanes.length + " branches", 0],
  ["Merges", D.merges.length, "leader → master", 0],
  ["Queued", D.wait.toLocaleString() + " s", D.worst ? "worst hold " + fmt(D.worst[1]) : "—", 0],
];
document.getElementById("stats").innerHTML = stats.map(s =>
  '<div class="stat' + (s[3] ? ' lead' : '') + '"><div class="k">' + s[0] +
  '</div><div class="v">' + s[1] + '</div><div class="n">' + s[2] + '</div></div>').join("");

// Ticks every 10 min, or every 30 for trials past three hours.
const step = D.span > 10800 ? 1800 : 600;
const grid = [];
for (let s = step - ((D.t0 % step) || 0); s < D.span; s += step) grid.push(s);

const ax = document.getElementById("axis");
grid.forEach(g => {
  const d = document.createElement("div");
  d.className = "tk"; d.style.left = pct(g) + "%";
  d.innerHTML = "<span>" + hhmm(g) + "</span>";
  ax.appendChild(d);
});

const lanesEl = document.getElementById("lanes"), tbody = document.getElementById("tbody");
D.lanes.forEach(t => {
  const hue = "var(" + t.var + ")";
  const mine = D.leases.filter(r => r[0] === t.id);
  const held = mine.reduce((a, r) => a + r[3], 0);
  const wait = mine.reduce((a, r) => a + r[2], 0);
  const cms = D.commits[t.id] || [];

  const lane = document.createElement("div");
  lane.className = "lane";
  lane.innerHTML = '<div class="rail"><div class="nm"><i class="chip" style="background:' + hue +
    '"></i>' + t.name + '</div><div class="sub">' + mine.length + ' leases · ' + held + 's</div></div>';

  const tr = document.createElement("div");
  tr.className = "track";
  const add = (cls, left, width, style, title) => {
    const d = document.createElement("div");
    d.className = cls; d.style.left = pct(left) + "%";
    if (width !== null) d.style.width = Math.max(pct(width), 0) + "%";
    Object.assign(d.style, style || {});
    if (title) d.title = title;
    tr.appendChild(d); return d;
  };
  t.cycles.forEach((c, i) => {
    const d = add("cyc" + (i % 2 ? " alt" : ""), c[0], c[1] - c[0]);
    d.innerHTML = '<span class="cl">c' + (i + 1) + '</span>';
  });
  grid.forEach(g => add("gl", g, null));
  D.merges.forEach(m => add("mrg", m[0], null, {}, m[1]));
  mine.forEach(r => {
    const [, s, w, h, label, st] = r;
    if (w > 0) add("blk wait", s, w, { background: hue }, label + " — queued " + fmt(w));
    add("blk" + (st ? " fail" : ""), s + w, h, { background: hue },
        label + " — held " + fmt(h) + (st ? " (exit " + st + ")" : ""));
  });
  cms.forEach(c => add("cm", c, null, { background: hue }, "commit " + hhmm(c)));

  lane.appendChild(tr); lanesEl.appendChild(lane);

  const row = document.createElement("tr");
  row.innerHTML =
    '<td><span class="chip" style="display:inline-block;background:' + hue +
    ';margin-right:7px;vertical-align:middle"></span>' + t.name + '</td>' +
    '<td class="n">' + mine.length + '</td><td class="n">' + held + ' s</td>' +
    '<td class="n">' + wait + ' s</td><td class="n">' + cms.length + '</td>' +
    '<td class="mono" style="font-size:12px">c' + t.cycles.length + ' running</td>';
  tbody.appendChild(row);
});

const idle = (100 - D.occ).toFixed(1);
document.getElementById("reading").innerHTML =
  '<div class="tag">How to read the occupancy figure</div><h3>' +
  (D.occ < 60 ? 'The GPU is idle ' + idle + '% of the wall clock' : 'The card is busy ' + D.occ + '% of the wall clock') +
  '</h3><p>' + D.lanes.length + ' agents on one serialized lease produced <b class="hl">' +
  fmt(D.held) + '</b> of held time in ' + fmt(D.span) + '. ' +
  (D.occ < 60
    ? 'What fills the gap is reading, deriving work models and writing kernels — so the card is <b class="hl">not</b> the binding constraint, and a second GPU would buy less than it appears.'
    : 'The card is close to saturated, so adding GPU capacity would plausibly convert into throughput.') +
  (D.worst ? ' The longest single hold was <span class="mono">' + D.worst[0] + '</span> at <b class="hl">' +
     fmt(D.worst[1]) + '</b>, against a median of ' + D.median + ' s.' : '') +
  '</p>';
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", default=".", help="trial dir (default: cwd)")
    ap.add_argument("--repo", default=None, help="repo root (default: derived from --trial)")
    ap.add_argument("--log", default=os.environ.get("CB_GPU_LOG", "/tmp/cuda-bench.gpu.log"))
    ap.add_argument("--out", default=None, help="output HTML (default: <trial>/timeline.html)")
    a = ap.parse_args()

    trial = os.path.abspath(a.trial)
    repo = a.repo or git(trial, "rev-parse", "--show-toplevel").strip() or trial
    out = a.out or os.path.join(trial, "timeline.html")

    d, path = build(trial, repo, a.log, out)
    print(f"{path}")
    print(f"  {d['span'] // 60} min · {len(d['leases'])} leases · {d['occ']}% occupancy "
          f"· {d['ncommit']} commits · {len(d['merges'])} merges")
    if d["worst"]:
        print(f"  longest hold: {d['worst'][0]} at {d['worst'][1]}s")


if __name__ == "__main__":
    main()

"""One profiled forward/step of the installed solution.py. TEMPLATE.

Warmup runs before the profiler is armed, so `ncu --profile-from-start off`
captures exactly one steady-state iteration. That matters for the reason the
playbook gives: without it `--launch-count 1` grabs whichever kernel launched
first, which on this path is a warmup, at cold clocks.

Inputs are built exactly as benchmark.py builds them — same seed — so the work
being profiled is the work being scored.

    ../run.sh py <NN> /path/to/prof_driver.py <graded-shape-index>

THE ARGUMENT IS AN INDEX INTO `shapes.SHAPES`, NOT A SIZE. This is deliberate,
and it replaces a convention that silently profiled the wrong thing for two
trials: the old template did `reference.T = ARG`, which is problem 03's own
knob. On every other problem that assignment either sets an attribute nothing
reads or sets one of several fields that define a shape, so the driver ran the
DEFAULT shape whatever you passed, and the .ncu-rep looked completely normal.
Task 02 lost every profile it took before cycle 2 to this.

Two rules keep the replacement honest, and both matter:

  * An index that is not a graded shape is a hard exit, never a fallback. A
    profiling driver that quietly substitutes a shape produces a report that is
    wrong in a way no downstream reader can detect.
  * Copy EVERY field of the matched shape onto the module, not just the one that
    varies today. A driver that copies only `T` goes silently wrong the first
    time a deck varies a second field.

TODO(task): problems 01 and 02 expose `Model.forward`; 03 exposes
`prefill`/`decode_steps` and 04 exposes `run(num_envs, horizon, seed)`. Replace
the marked block with this problem's own construction and call, copied from its
benchmark.py rather than guessed at.
"""
import sys

import torch

sys.path.insert(0, ".")
import reference  # noqa: E402
import shapes  # noqa: E402
import solution  # noqa: E402

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WARMUP = 5

if not 0 <= IDX < len(shapes.SHAPES):
    sys.exit(f"prof_driver: shape index {IDX} out of range; "
             f"this problem grades {len(shapes.SHAPES)} shapes (0..{len(shapes.SHAPES) - 1})")
shape = shapes.SHAPES[IDX]

dev = torch.device("cuda:0")

# --- TODO(task): build the model and inputs exactly as benchmark.py does -----
# Every field, not just the one that varies today — see the module docstring.
for field, value in shape.items():
    setattr(reference, field, value)

model = solution.Model(*reference.get_init_inputs()).to(dev).eval()
torch.manual_seed(2026)
inputs = [t.to(dev) for t in reference.get_inputs()]


def one_iteration():
    model(*inputs)
# ---------------------------------------------------------------------------

with torch.no_grad():
    for _ in range(WARMUP):
        one_iteration()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    one_iteration()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

print(f"profiled one iteration at graded shape {IDX}: {shape}", flush=True)

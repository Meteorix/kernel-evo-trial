"""Fast correctness probe for the installed solution.py — seconds, not minutes.

check.py runs the full shape sweep x seeds x numeric-stress cases against the
reference, which is minutes per run. This does the same comparison at a few
small shapes so a layout or indexing bug shows up before paying for the real
gate. Passing this is necessary, never sufficient: `../run.sh check <NN>` is the
gate, and only its PASS counts for a promotion.

This does not hold the GPU lease. It launches kernels, so wrap it when another
worker may be measuring:

    ../gpu.sh smoke-<NN> -- ../run.sh py <NN> /path/to/smoke.py [args ...]

TODO(task): set TOL and the stress scales from this problem's own problem.yaml /
check.py — do not copy 01's. Tolerances differ across the deck (01 and 03 use
0.08, 02 uses 0.1, 04 needs logits ~1e-3 with positions and rewards **exact**),
and inventing one is exactly the failure playbook rule 1 exists to prevent.
"""
import sys

import torch

sys.path.insert(0, ".")
import reference  # noqa: E402
import solution  # noqa: E402

ARGS = [int(a) for a in sys.argv[1:]] or [1, 8, 65, 256]
dev = torch.device("cuda:0")
TOL = 0.08                                    # TODO(task): from problem.yaml
SCALES = ((1.0, "nominal"), (1e-2, "small"), (8.0, "large"))   # TODO(task)

worst = 0.0
for arg in ARGS:
    # --- TODO(task): construct reference and solution for this problem ------
    reference.T = arg
    init = reference.get_init_inputs()
    ref = reference.Model(*init).to(dev).eval()
    sol = solution.Model(*init).to(dev).eval()
    sol.load_state_dict(ref.state_dict(), strict=True)

    for scale, label in SCALES:
        torch.manual_seed(42)
        inputs = [t.to(dev) for t in reference.get_inputs()]
        inputs[0] = (inputs[0].float() * scale).half()
        with torch.no_grad():
            r, o = ref(*inputs), sol(*inputs)
    # -----------------------------------------------------------------------

        d = (r.float() - o.float()).abs()
        allowed = TOL + TOL * r.float().abs()
        bad = int((d > allowed).sum())
        worst = max(worst, float((d / allowed).max()))
        print(
            f"arg={arg:<6} {label:<8} max_abs={float(d.max()):9.5f} "
            f"ref_max={float(r.float().abs().max()):8.3f} "
            f"bad={bad}/{d.numel()} {'ok' if bad == 0 else 'FAIL'}"
        )

print(f"worst error/allowed ratio = {worst:.3f}  ({'ok' if worst <= 1 else 'FAIL'})")

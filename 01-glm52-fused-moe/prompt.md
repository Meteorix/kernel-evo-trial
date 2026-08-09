# 01-glm52-fused-moe — the task

Verbatim from `kernelbench.com/benchmarks/cuda/problems-rtx2070/01_*/PROMPT.txt` at the
pinned bench commit. The scoring rules, shapes and tolerances live beside it in
`problem.yaml`, `shapes.py`, `check.py` and `benchmark.py` — read those too, and
trust them over anything summarised here.

Then read `KDA.md` for how to attack it and `SWARM.md` for how to work alongside
the other three workers.

---

I need a fast fused MoE layer in CUDA for the RTX 2070 (SM75 Turing, GDDR6, 448 GB/s, 8 GB), matching the GLM-5.2 MoE structure. Reference is reference.py; answer in solution.py with the same Model interface.

GLM-5 / 5.1 / 5.2 MoE layer (not Mixtral):
  - E=128 routed experts, top_k=8
  - n_shared=1 shared expert that ALWAYS runs on every token
  - H=2048, I=1024 (moe intermediate)

Fused weight layout (vLLM-style packing, GLM routing):
  w1_routed: (E, 2*I, H) fp16 — [0:I)=gate, [I:2I)=up
  w2_routed: (E, H, I) fp16
  w1_shared: (n_shared, 2*I, H), w2_shared: (n_shared, H, I)
Per expert: h = silu(x @ gate.T) * (x @ up.T); y = h @ down.T
out = sum_shared(y_s) + sum_k weight[t,k] * y_routed[e_k]

Inputs: x (T,H) fp16, expert_ids (T,top_k) int64, expert_weights (T,top_k) fp16.
Routing is given. Match reference within ~0.08 on fp16.

Shapes in shapes.py are a serving-style geomean sweep (Hard FP8 technique):
aligned T=2048, T=2079 misaligned pack tails, T=1 decode microbatch, T=4096
prefill, T=512, T=1000 off-alignment. Do not hyperspecialize to one T.

This is KernelBench-CUDA: real CUDA only (load_inline / .cu / CUTLASS C++).
Forbidden: Triton, import vllm, flashinfer MoE, megablocks, torch._grouped_mm.
You may read vLLM fused_moe for packing ideas; reimplement in CUDA — do not call it.

Flywheel: implement, profile, benchmark.py, `python check.py`. check.py caps T at
256; benchmark uses full shapes. If check.py has not printed PASS, you are not
done. Take as long as you need.

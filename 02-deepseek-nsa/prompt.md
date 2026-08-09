# 02-deepseek-nsa — the task

Verbatim from `kernelbench.com/benchmarks/cuda/problems-rtx2070/02_*/PROMPT.txt` at the
pinned bench commit. The scoring rules, shapes and tolerances live beside it in
`problem.yaml`, `shapes.py`, `check.py` and `benchmark.py` — read those too, and
trust them over anything summarised here.

Then read `KDA.md` for how to attack it and `SWARM.md` for how to work alongside
the other three workers.

---

I need a fast DeepSeek NSA-inspired sparse attention kernel in CUDA for the RTX 2070 (SM75 Turing, GDDR6, 448 GB/s, 8 GB). Reference is reference.py; answer in solution.py with the same Model interface.

Semantics (bench-simplified Native Sparse Attention — same as reference.nsa_attend):
  Inputs q,k,v are (B, H, S, D) fp16. Causal. For each query position t:
  1. Split keys into blocks of block_size=64.
  2. Block importance = mean of (q·k / sqrt(D)) over keys in the block that are causal (j<=t).
  3. Select the top_n_blocks=8 blocks by importance, union with a local sliding window of the last sliding_window=64 tokens, causal only.
  4. Softmax attention over the selected key indices only; write o[t].

Match the reference within ~0.1 abs/rel on fp16. Shapes in shapes.py are a
geomean sweep (Hard-style): aligned S, S not multiple of block_size=64 (e.g.
2079, 4095), long context, batched short. FLOPs are dense-equivalent (full S×S)
so sparsity does not inflate TFLOPS — speed comes from doing less work well.


This is KernelBench-CUDA: real CUDA only. Forbidden: Triton, flash-attn, flashinfer, SDPA, CuteDSL. You may read the NSA paper (arXiv 2502.11089) and DeepSeek materials; reimplement the *bench* semantics in reference.py (not a private production bit-exact path).

The hard part is fusing block scoring, top-n select, gather, and online sparse attention — not pasting a dense FlashAttention tutorial.

Flywheel: implement, profile, benchmark.py, `python check.py`. check.py uses smaller S for wall-clock; benchmark uses full shapes. If check.py has not printed PASS, you are not done. Take as long as you need.

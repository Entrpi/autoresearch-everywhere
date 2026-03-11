# Kernel Lab

The kernel lab is the repo's place for backend-specific kernel work under one top-level workflow.

The current shape is intentionally modest:

- top-level entrypoint: `kernel-lab.py`
- shared boundary: `autoresearch_lab/`
- first implementation: MLX under `autoresearch_mlx/`
- future implementations: Triton/CUDA, ROCm, ANE

The pattern comes from the same idea behind `autokernel`, but adapted to this repo:

- keep one mutable workspace file
- keep the benchmark fixed
- optimize one target kernel at a time
- only later lift successful backend-specific labs into a shared cross-engine story

## Current MLX Flow

List the current MLX targets:

```bash
uv run kernel-lab.py --engine mlx list-targets
```

Create a workspace for the starter-ready target:

```bash
uv run kernel-lab.py --engine mlx init --target rmsnorm --workspace /tmp/mlx-rmsnorm-lab
```

Run the fixed bench harness against that workspace:

```bash
uv run kernel-lab.py --engine mlx bench --workspace /tmp/mlx-rmsnorm-lab
```

For now the MLX lab is deliberately narrow:

- starter-ready targets:
  - `rmsnorm`
  - `layernorm`
  - `rmsnorm_backward`
  - `layernorm_backward`
  - `residual_blend`
  - `residual_rmsnorm`
  - `qk_rmsnorm`
  - `rope_qk_fused`
  - `logits_softcap`
  - `activation_pointwise`
  - `rotary_embedding`
  - `reduce`
  - `softmax`
  - `value_embed_gate`
  - `attention_mask_local`
  - `cross_entropy_prelude`
  - `fused_mlp`
- `flash_attention` is explicitly deferred as a first target because MLX already has optimized attention primitives and custom backward there is a worse place to start

## Why A Shared Lab Boundary

The goal is not to force all backends into one kernel API.

The goal is to give them the same outer workflow:

- list candidate targets
- create a mutable workspace
- run a fixed benchmark harness
- later add profiling, extraction, orchestration, and promotion on top

That is why the current shared package is `autoresearch_lab/`, while the actual kernel mechanics stay backend-specific.

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

The important split now is between:

- a heuristic layer that is fast and useful for choosing candidates
- a trace layer that records real Metal captures and acts as the truth source for Apple Silicon performance work

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

Verify a workspace before promotion:

```bash
uv run kernel-lab.py --engine mlx verify --workspace /tmp/mlx-rmsnorm-lab --quick
```

Rank likely next targets for a preset:

```bash
uv run kernel-lab.py --engine mlx profile --preset m5-balanced --top-k 8 --output /tmp/mlx-profile.json
```

Turn a profile into the next ready workflow:

```bash
uv run kernel-lab.py --engine mlx orchestrate --profile /tmp/mlx-profile.json --workspace-root /tmp/mlx-lab
uv run kernel-lab.py --engine mlx extract --profile /tmp/mlx-profile.json --workspace /tmp/mlx-lab/block-pipeline --rank 1
```

Capture a real Metal trace for a workspace:

```bash
uv run kernel-lab.py --engine mlx capture --workspace /tmp/mlx-lab/block-pipeline --output /tmp/mlx-lab/block-pipeline.gputrace --quick
```

That writes:

- a `.gputrace` bundle you can open in Xcode
- a `.metadata.json` sidecar with the bench result, device info, and capture context
- a persistent evidence entry in `results/kernel_lab/ledger.jsonl`

For now the MLX lab is deliberately narrow, but it is no longer just `init + bench`:

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
  - `ve_lookup_reshape`
  - `attention_mask_local`
  - `proj_head_reshape`
  - `loss_logits_cast_softcap`
  - `cross_entropy_prelude`
  - `cross_entropy_full`
  - `attention_prelude`
  - `block_prelude`
  - `fused_mlp`
- `flash_attention` is explicitly deferred as a first target because MLX already has optimized attention primitives and custom backward there is a worse place to start
- first orchestration layer:
  - `profile` emits ranked candidate targets for a preset using model-aware heuristics
- `extract` creates a starter workspace from a profile artifact and preserves the profile context
- `orchestrate` emits the next ready command sequence for a selected ranked target
- `verify` is the fixed-harness promotion gate above a quick bench
- `capture` records a real MLX Metal trace plus metadata for a workspace run

If you already captured a trace for a candidate workspace, pass its metadata back into orchestration:

```bash
uv run kernel-lab.py --engine mlx orchestrate --profile /tmp/mlx-profile.json --workspace-root /tmp/mlx-lab --rank 1 --trace-metadata /tmp/mlx-lab/block-pipeline.metadata.json
```

That upgrades the plan from "interesting candidate" to "trace-backed target" and adds the trace review step to the suggested workflow.

If a target has already been both verified and captured, orchestration will reuse the last known workspace and upgrade the plan again to `promotion-ready`, which means the next recommended step is an end-to-end integration A/B rather than another blank workspace.

## Heuristic Layer vs Trace Layer

The current MLX lab deliberately uses two different kinds of evidence.

The heuristic layer is:

- `profile`
- `extract`
- `orchestrate`
- `bench`
- `verify`

It also now includes a persistent evidence ledger. Heuristic profiles are no longer stateless: they can see whether a target already has successful `verify` and `capture` events for the same preset and backend family.

Use it constantly. It is fast, cheap, and good at deciding what to try next.

The trace layer is:

- `capture`
- Xcode Metal Debugger
- Metal System Trace / GPU counters once a capture is open

Use it when a target stops being "interesting" and starts being "worth believing."

The rule of thumb is:

- heuristics choose candidates
- the ledger remembers what has already been proved
- captures validate reality

That matters on Apple Silicon because a synthetic microbench can miss the real cost of synchronization, hidden copies, queue pacing, or other host/device effects.

## Why A Shared Lab Boundary

The goal is not to force all backends into one kernel API.

The goal is to give them the same outer workflow:

- list candidate targets
- create a mutable workspace
- run a fixed benchmark harness
- profile end-to-end opportunities
- extract the next target into a workspace
- orchestrate the next kernel-lab step
- verify candidate work before promotion
- capture trace artifacts when the backend supports them

That is why the current shared package is `autoresearch_lab/`, while the actual kernel mechanics stay backend-specific.

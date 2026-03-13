# Kernel Lab

Kernel-lab is the repo's workshop for trying low-level speedups safely.

Modern training runs spend time in lots of small repeated operations: normalizing activations, reshaping tensors, applying rotary embeddings, preparing attention inputs, and reducing loss-side values. Some of those are good candidates for custom kernels, but dropping kernel experiments straight into the main trainer is risky. A candidate can be correct but irrelevant, fast in isolation but useless end to end, or only beneficial on one machine.

Kernel-lab exists to separate:

- "this looks like a promising low-level optimization"
- "this is proven enough to earn a place in the real training path"

So the workflow is deliberately staged:

- profile likely targets
- create a small mutable workspace for one target
- benchmark and verify it in isolation
- capture a real backend trace when needed
- then test it against the real trainer before considering promotion

That story is shared across backends even though the actual kernel substrate differs. Metal kernels on Apple GPUs, Triton/CUDA kernels on NVIDIA, HIP/ROCm kernels on AMD, and future accelerator-specific paths can all plug into the same outer workflow.

If you already know tools like CUTLASS, Triton, CK, or rocWMMA, the easiest way to think about kernel-lab is: it is not another kernel library or compiler layer. It sits one layer above those systems. Those tools are how a backend-specific kernel gets implemented; kernel-lab is how the repo decides which kernel opportunities are worth pursuing, how they are benchmarked, how trace evidence is collected, and what it takes to promote them into the actual training engine.

The current shape is intentionally modest:

- top-level entrypoint: `kernel-lab.py`
- shared boundary: `autoresearch_lab/`
- first deep implementation: MLX under `autoresearch_mlx/`
- first non-MLX expansion: CUDA trace automation under `autoresearch_cuda/`
- future implementations: Triton workspaces on CUDA, ROCm, ANE

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

Inspect the persistent evidence for one target:

```bash
uv run kernel-lab.py --engine mlx evidence --target block_prelude --preset m5-balanced
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

Run a real end-to-end trainer A/B for a directly integrated target:

```bash
uv run kernel-lab.py --engine mlx integration-ab --workspace /tmp/mlx-lab/block-pipeline --time-budget 20 --benchmark-skip-eval --no-checkpoint
```

If `--preset` is omitted, the lab uses the calibrated platform default for the current device. That is the normal path once `calibrate.py` has been run on the machine. Pass `--preset` explicitly only when you are doing a smoke check or intentionally targeting a non-default operating point.

`integration-ab` now uses repeated balanced measured rounds by default (`--repeats 2`) after warmup, so promotion is based on repeated end-to-end evidence rather than one encouraging run.

When you want stronger evidence, use the suite form:

```bash
uv run kernel-lab.py --engine mlx integration-suite --workspace /tmp/mlx-lab/block-pipeline --time-budget 20 --benchmark-skip-eval --no-checkpoint
```

If `--preset` is omitted, `integration-suite` uses the calibrated platform default and the next stronger preset by default. This is the preferred promotion path once a target is past smoke checks, because it tests whether the signal survives beyond one operating point.

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

For the current MLX lab, only a subset of targets has a direct trainer-side integration hook. Narrow path targets such as `logits_softcap`, `rotary_embedding`, `value_embed_gate`, `attention_prelude`, and `fused_mlp` can already run through `integration-ab` and `integration-suite`. Broader composed targets like `block_prelude` can still be profiled and traced, but they will surface as `needs-integration-adapter` until a direct training-path hook exists.

You can ask for that decision directly:

```bash
uv run kernel-lab.py --engine mlx promotion-check --target block_prelude
```

Once `integration-ab` has run, the ledger can now distinguish:

- `integration-validated`
- `integration-regressed`
- `integration-mixed`

So promotion is no longer “trace-backed forever.” It can advance, stall, or back off based on repeated trainer-side A/B evidence with pair counts, cross-preset coverage, and relative effect sizes, not just one baseline/candidate pair.

## CUDA Trace Automation

CUDA now has the first trace-first kernel-lab path, plus a narrow first starter-workspace layer for the most traceable target families.

The point of this first CUDA slice is different from the MLX slice:

- MLX proves the end-to-end lab workflow, but its serious trace tooling is still GUI/manual
- CUDA is where the repo can start automating trace review in earnest

That trace-first CUDA path is now grounded on a real GB10 system, not just synthetic artifacts. The first full Blackwell GB10 run validated:

- real trainer smoke with installed FlashAttention via [FA4 PR](https://github.com/Dao-AILab/flash-attention/pull/2268)
- real Nsight Systems capture
- real `trace-profile` / `auto-review`
- real Nsight Compute `deep-profile` once host GPU counters were enabled
- real starter workspace `bench` / `verify`

The concrete environment for that validation was eugr's vLLM container image, tracked as the local March 1, 2026 snapshot of:

- `vllm-node-tf5:latest`
- image ID `sha256:c1ba011f841cacdfc234e5b754b1cb5e8120b8d4bd6b896c6703b28a44ba185a`
- source repo: [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
- best available source pin for that build date: `8f11e7e5edd8c964f7a44fbd29f0c86a8df49a82` (the repo commit at HEAD before the local image creation timestamp)
- NVIDIA PyTorch `26.01` image family (`com.nvidia.build.id=256811084`, `com.nvidia.build.ref=9fa5c48351cf93ac6e6972ca113a7e3c54675a76`)
- PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`
- CUDA runtime `13.1`

with the repo bind-mounted into `/workspace/autoresearch-everywhere`.

The first grounded GB10 trace currently says:

- dominant issue: `sync-bound`
- `launch_fusion` accounts for about `65.4%` of observed CUDA kernel time
- `data_movement` accounts for about `34.6%`

The first deeper diagnoses are both still weak or mixed, which is why the lab keeps promotion conservative:

- `launch_fusion` can move into a real CUDA workspace
- `data_movement` has a correct workspace and real GB10 bench/verify results, but still lands in `needs-manual-cuda-review` because the deeper diagnosis is not strong enough yet

So the first CUDA lab commands are:

```bash
uv run kernel-lab.py --engine cuda list-targets
uv run kernel-lab.py --engine cuda capture --preset upstream --time-budget 20 --output /tmp/cuda-upstream-trace
uv run kernel-lab.py --engine cuda trace-profile --metadata /tmp/cuda-upstream-trace.metadata.json --output /tmp/cuda-upstream-trace.profile.json
uv run kernel-lab.py --engine cuda auto-review --trace-profile /tmp/cuda-upstream-trace.profile.json
uv run kernel-lab.py --engine cuda evidence --target launch_fusion --preset upstream
uv run kernel-lab.py --engine cuda orchestrate --trace-profile /tmp/cuda-upstream-trace.profile.json --workspace-root /tmp/cuda-lab
uv run kernel-lab.py --engine cuda promotion-check --target launch_fusion --preset upstream
uv run kernel-lab.py --engine cuda extract --profile /tmp/cuda-upstream-trace.profile.json --workspace /tmp/cuda-lab/launch_fusion --rank 1
uv run kernel-lab.py --engine cuda bench --workspace /tmp/cuda-lab/launch_fusion --device cuda --quick
uv run kernel-lab.py --engine cuda verify --workspace /tmp/cuda-lab/launch_fusion --device cuda --quick
```

That flow currently does seven things:

- `capture`
  - runs the real CUDA trainer under Nsight Systems
  - writes a `.nsys-rep`
  - writes a metadata sidecar that records the trainer command, tool versions, and report artifacts
- `trace-profile`
  - parses the machine-readable Nsight report exports
  - groups kernel names into target families such as `flash_attention`, `fused_mlp`, `norm`, `data_movement`, or `launch_fusion`
  - ranks them by observed time share plus simple bottleneck-aware heuristics
- `auto-review`
  - classifies the run as `launch-bound`, `sync-bound`, `copy-bound`, `kernel-dominated`, or `mixed`
  - records that judgment into the shared ledger as machine-generated evidence
- `deep-profile`
  - optionally reruns the top trace-ranked family under Nsight Compute
  - records a deeper diagnosis such as `compute-bound`, `bandwidth-bound`, or `under-occupied`
  - is the next step when a starter-ready family looks important enough that Nsight Systems timing alone is no longer enough
  - strong diagnoses now boost later ranking/promotion confidence, while weak or mixed ones block promotion pending manual CUDA review
- `evidence`
  - summarizes what the ledger currently knows about one CUDA target on one preset
  - exposes whether trace review is still thin, already trace-backed, ready for a starter workspace, or currently deprioritized
- `orchestrate`
  - consumes a trace-profile artifact plus accumulated evidence
  - picks the next CUDA target family to pursue
  - for starter-ready families, emits real `extract` / `bench` / `verify` commands
  - for broader families, still acts as a planning layer for future CUDA/Triton workspace work
- `extract`
  - instantiates a starter CUDA workspace from a ranked trace-profile result when the family has a fixed harness
- `bench` / `verify`
  - run the fixed starter harness for that workspace
  - default to CUDA on NVIDIA machines, but degrade cleanly if `torch` or CUDA is missing
- `promotion-check`
  - tells you whether a target still needs capture, needs structured trace review, is ready for a starter workspace, is ready for a future workspace family, or is currently deprioritized

Starter CUDA workspaces currently exist for:

- `launch_fusion`
- `norm`
- `logits_softcap`
- `loss_prelude`
- `data_movement`
- `matmul_epilogue`
- `attention_prelude`
- `value_embed_gate`
- `rope_qk_fused`
- `fused_mlp`

The first Triton-backed CUDA workspace slice is intentionally narrow:

- `launch_fusion` now ships with an optional Triton residual-add kernel
- `norm` now ships with an optional Triton RMSNorm kernel
- `loss_prelude` now ships with an optional Triton row-wise cross-entropy-prelude kernel
- `logits_softcap` now ships with an optional Triton pointwise softcap kernel
- `value_embed_gate` now ships with an optional Triton pointwise gate-application kernel
- `rope_qk_fused` now ships with an optional Triton row-wise RoPE + RMSNorm kernel
- `fused_mlp` now ships with an optional Triton pointwise squared-ReLU activation kernel
- `attention_prelude` now ships with an optional Triton gate-application kernel inside the broader Q/K/V staging path
- `data_movement` now ships with an optional Triton copy/reshape kernel
- `matmul_epilogue` now ships with an optional Triton matmul+bias kernel
- the CUDA starter catalog is still intentionally narrow: each Triton-backed target accelerates one honest seam inside the larger path rather than pretending the whole surrounding block is already a Triton-native rewrite

`norm`, `logits_softcap`, `loss_prelude`, `matmul_epilogue`, `fused_mlp`, `attention_prelude`, `value_embed_gate`, `rope_qk_fused`, `launch_fusion`, and `data_movement` currently have direct CUDA trainer-side hooks. They are the CUDA targets that can now move past workspace-local evidence into real `integration-ab` and `integration-suite` runs.

## CUDA profiling setup notes

For full CUDA profiling, the host must allow GPU performance counters. If `deep-profile` fails with `ERR_NVGPUCTRPERM`, the usual Linux/NVIDIA fix is:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/99-nvidia-profiling.conf
sudo update-initramfs -u
sudo reboot
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

After reboot, `RmProfilingAdminOnly: 0` means the host is configured correctly. In our container setup, adding `--cap-add=SYS_ADMIN` to the Docker run also helped ensure `ncu` could collect metrics cleanly.

The GB10 runs here were executed in the `vllm-node-tf5:latest` container, roughly like:

```bash
docker run --gpus all \
  --cap-add=SYS_ADMIN \
  --ipc=host \
  --shm-size=16g \
  --rm \
  -v /home/ent/autoresearch-everywhere:/workspace/autoresearch-everywhere \
  -v /home/ent/.cache/autoresearch:/root/.cache/autoresearch \
  -w /workspace/autoresearch-everywhere \
  vllm-node-tf5:latest ...
```

One practical note from the GB10 validation: newer Nsight Compute CSV exports can come back in a wide-table format instead of the older long `Metric Name` / `Metric Value` format. The CUDA lab now supports both, because the first real GB10 `deep-profile` surfaced exactly that parser mismatch.

The intended first CUDA trainer-side loop is:

```bash
uv run kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-lab/loss_prelude --preset upstream --time-budget 20 --benchmark-skip-eval --no-checkpoint
uv run kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-lab/loss_prelude --preset upstream --time-budget 20 --repeats 2 --benchmark-skip-eval --no-checkpoint
```

If PyTorch or CUDA is not installed on the current machine, those commands return structured `missing-runtime` results instead of failing as opaque shell errors. That keeps the outer CUDA lab workflow stable even on development machines that cannot execute the full trainer-side path yet.

If Nsight is not installed, the CUDA commands still emit structured metadata and explicit fallback statuses (`missing-tool`, `capture-unavailable`, `insufficient-trace-data`) instead of failing as an opaque shell error. That makes it possible to keep the outer workflow stable across developer machines that do not yet have NVIDIA tooling installed.

This is the inverse of the MLX constraint:

- on MLX, capture is scriptable but serious review is still mostly manual
- on CUDA, the long-term goal is for capture and first-pass review to be scriptable by default, with Nsight GUI inspection as the escalation path

That is why the current CUDA lab starts from traceability rather than from Triton workspaces. The first thing worth proving on NVIDIA is that the repo can discover and classify real kernel opportunities automatically before it starts minting backend-specific workspaces. The starter workspaces now exist to turn that trace-backed prioritization into a concrete next step for a few narrow families, not to replace the trace-first logic.

## Heuristic Layer vs Trace Layer

The current MLX lab deliberately uses two different kinds of evidence, and CUDA is starting from the trace-heavy half of that split.

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
- `review-trace` on MLX
- `trace-profile` and `auto-review` on CUDA
- Xcode Metal Debugger
- Metal System Trace / GPU counters once a capture is open
- Nsight Systems, and later Nsight Compute, on CUDA

Use it when a target stops being "interesting" and starts being "worth believing."

On MLX, `review-trace` is the bridge between the Xcode-facing world and the orchestration loop. It records whether the captured target looked `high`, `medium`, `low`, or `none` in end-to-end relevance, so later profiles can both boost and demote targets instead of only rewarding the existence of a capture.

On CUDA, the analogous bridge is now `trace-profile` plus `auto-review`, with `evidence`, `orchestrate`, and `promotion-check` layered on top. The trace is captured from a real trainer run, summarized into target families automatically, classified into a bottleneck mode, and then fed back into the same “what should we optimize next?” loop before the repo asks a human to look at a GUI.

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

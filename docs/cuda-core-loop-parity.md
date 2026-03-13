# CUDA Core-Loop Parity

This note evaluates what it will take to bring the CUDA path up to feature parity with the current MLX-first core trainer loop.

The question is no longer whether CUDA can run at all. It can:

- `prepare.py --engine cuda` exists
- `train.py --engine cuda` exists
- `calibrate.py --engine cuda` exists
- `kernel-lab.py --engine cuda` exists
- the CUDA path has real architecture-family awareness
- the CUDA lab now has real end-to-end trace automation
- the GB10 path has already been validated against a real Blackwell system

The real question is this:

- what still separates CUDA from the MLX path as a full autoresearch engine?
- which gaps are architectural, and which are just unimplemented?
- how should GB10, A100, H100, and B200 each be used to close those gaps?

## Executive Summary

CUDA is already beyond the original upstream path in several important ways:

- it runs behind the same top-level front doors as MLX
- it has a real engine boundary
- it has hardware-family-aware runtime policy
- it has a real kernel-lab path with trace automation
- it now runs successfully on GB10 with FlashAttention 4 from the SM120-support PR

But CUDA is still behind MLX on the core autoresearch loop itself.

The largest remaining gaps are:

1. no runtime eval-policy ladder comparable to MLX cheap/reference/full
2. no candidate-default promotion path through `calibrate.py --engine cuda`
3. no CUDA trainer integration evidence yet for kernel-lab targets
4. no broad hardware validation matrix yet beyond GB10

One important nuance from the latest GB10 work is that "longer short probes" are not the same thing as real horizon correction. The current fast-mode bring-up can now normalize hardware batch shape before family comparison, but the repo still does not have enough true intermediate `val_bpb` data to predict a `300s` winner reliably from very short runs. The next corrective step is therefore not another heuristic rerank; it is periodic validation checkpoints during longer CUDA runs so the repo can build a true horizon dataset from real `300s` trajectories.

So the shortest honest summary is:

- CUDA is already a serious training engine and a serious kernel-lab backend
- CUDA now has exact checkpoints, exact resume, and checkpoint-backed rung evaluation on GB10
- CUDA is not yet at MLX parity for the full calibrated autoresearch loop

## What "Parity" Means Here

For this repo, feature parity on the core trainer loop does not mean "same code."

It means the CUDA engine can participate in the same important user stories:

1. A new user clones the repo on NVIDIA hardware and can:
   - run `prepare.py`
   - run `train.py`
   - run `calibrate.py`
   - get a sensible candidate default
   - understand where the machine sits between the local-laptop and upstream/H100 reference points

2. The runtime can make conservative decisions on its own:
   - known hardware behaves like a known platform
   - unknown hardware falls back visibly
   - stale or weak calibration is not silently over-trusted

3. Kernel-lab can:
   - trace the real trainer
   - prioritize real kernel families
   - validate targets against the real trainer
   - eventually promote validated targets into the live CUDA trainer path

4. The engine can survive meaningful stack changes:
   - new attention backend
   - new batching strategy
   - new compile path
   - new kernels
   - and then be recalibrated with one shared workflow

## Current State

### Already at or near parity

These pieces are already strong:

- Shared top-level entrypoints
  - `prepare.py --engine cuda`
  - `train.py --engine cuda`
  - `calibrate.py --engine cuda`
  - `kernel-lab.py --engine cuda`

- Shared training-engine boundary
  - `autoresearch_platform/engines.py`
  - `autoresearch_platform/cuda_engine.py`

- CUDA hardware-family detection and policy framing
  - A100-class Ampere
  - Ada RTX 40xx
  - Ada L40S-class
  - H100/Hopper
  - RTX 50xx-class consumer Blackwell
  - B200-class Blackwell
  - GB10 / DGX Spark
  - anticipated Vera Rubin placeholder

- CUDA trainer viability
  - real GB10 smoke succeeded
  - FlashAttention 4 from the SM120-support PR worked on GB10
  - fallback path now exists when the preferred FA path is unavailable

- CUDA kernel-lab maturity
  - `capture`
  - `trace-profile`
  - `auto-review`
  - `deep-profile`
  - starter Triton-backed workspaces
  - real GB10 trace evidence

### Still materially behind MLX

These are the real parity gaps:

| Area | MLX | CUDA | Gap |
| --- | --- | --- | --- |
| Platform bring-up | full | partial | CUDA now has local search, exact checkpoint minting/resume, and checkpoint-backed eval calibration, but still lacks promoted defaults in the shared engine |
| Checkpointing | full exact + async variants | exact sync | medium gap |
| Resume | full | exact sync | medium gap |
| Eval calibration | cheap/reference/full ladder plus runtime auto-selection | checkpoint-backed rung runner validated on GB10 | medium gap |
| Runtime eval policy | confidence/freshness-aware | none | major gap |
| Local search in bring-up | yes | yes | low gap |
| Train/wall benchmarking sophistication | high | lower | medium gap |
| Kernel-lab trace review | manual on MLX, but real | automated on CUDA | CUDA is actually ahead here |
| Kernel-lab trainer integration | partial on MLX | not yet real on CUDA | major gap |
| Hardware validation breadth | M5 deep | GB10 real, others pending | major gap |

The key takeaway is that CUDA is not generally behind everywhere. It is specifically behind on the **calibrated trainer-loop features**, while already competitive or ahead on parts of the **kernel-lab evidence stack**.

### Horizon data is still weaker than it should be

The recent GB10 calibration work also exposed a more subtle parity gap: the CUDA bring-up flow can now tune hardware batch shape early, but it still does not have a trustworthy way to infer the `300s` winner from short family probes alone.

What is true today:

- the trainer reports a final held-out `val_bpb`
- short bring-up probes can compare families at a shared tuned hardware batch
- long `300s` A/B runs already showed that the true best family can differ from what a short rerank suggests

What was missing until now:

- periodic validation checkpoints during the long run
- a machine-readable curve artifact that records `val_bpb` at multiple training horizons
- a small reporting tool that can compare those curves directly

That means the current "horizon correction" logic should still be treated as provisional. The repo now has the first real CUDA curve path:

- `autoresearch_cuda/train.py` can emit periodic validation checkpoints via `--curve-eval-seconds`
- it can write a JSON curve artifact with `--curve-output`
- `tools/cuda_curve_report.py` can summarize those artifacts for a target horizon

This is the foundation for replacing the current fake horizon rerank with a curve-backed model built from real `300s` trajectories.

## GB10: What We Already Proved

GB10 is the first real CUDA validation machine in this repo. It matters because it tested more than "can CUDA import?"

What is now grounded on GB10:

- CUDA trainer runs successfully
- FlashAttention 4 works from the SM120-support branch / PR
- exact sync checkpoint minting works through the core CUDA trainer loop
- exact resume works and restores loader position deterministically
- checkpoint-backed eval calibration works through the shared CUDA engine
- `resolved_attention_backend` is correctly surfaced in trainer output
- CUDA Nsight Systems trace capture works
- CUDA Nsight Compute deeper profiling works when host counters are enabled
- CUDA kernel-lab trace automation is real, not placeholder
- CUDA trace-family matching was corrected against real Blackwell output

What GB10 tells us:

- the shared engine/lab design is real enough to survive contact with actual NVIDIA hardware
- CUDA is not blocked on architecture awareness anymore
- the remaining parity work is mostly runtime policy, default promotion, and trainer-loop depth, not "bring-up impossibility"

What GB10 does **not** yet prove:

- parity on upstream/H100-style throughput or quality
- parity on runtime eval-policy auto-selection
- parity across the broader NVIDIA fleet

### GB10 eval calibration snapshot

The first real checkpoint-backed CUDA eval ladder on GB10 was run against an exact sync checkpoint from the `upstream` preset using the shared `CUDAEngine.run_eval_calibration(...)` path. The rung results were:

| Rung | Seq len | Batch | Eval tokens | `val_bpb` | Eval seconds | Abs error vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cheap` | `2048` | `32` | `262144` | `2.212186` | `1.0` | `0.038056` |
| `reference` | `2048` | `32` | `1572864` | `2.161972` | `5.5` | `0.012158` |
| `full` | `2048` | `32` | `20971520` | `2.174130` | `72.2` | `0.000000` |

Artifacts:

- markdown report: `/home/ent/autoresearch-everywhere/results/analysis/gb10_eval_calibration.md`
- per-rung logs under `/home/ent/autoresearch-everywhere/results/analysis/cuda_resume_smoke/eval_calibration_logs/`

One nuance: this exact rung-validation pass ran in the minimal `vllm-node-tf5:latest` container plus the extra tokenizer/data dependencies needed for eval-only mode, so it resolved `torch-sdpa` rather than the separately validated FlashAttention 4 path. That still proves the rung runner and engine flow; it just should not be confused with the earlier FA4-enabled trainer smoke.

### GB10 FA4 fast bring-up snapshot

After the FA4/SM120 path was installed from the FlashAttention PR branch and the compiled-checkpoint key mismatch was fixed, the full fast bring-up flow completed end to end on GB10 in the FA4-enabled container.

Final candidate default:

| Field | Value |
| --- | --- |
| Preset | `m5-small` |
| `seq_len` | `512` |
| `window_pattern` | `L` |
| `device_batch_size` | `32` |
| `total_batch_size` | `32768` |
| `grad_accum_steps` | `2` |
| ranking `val_bpb` | `1.291656` |
| ranking steady tok/s | `541924.6` |

Measured zones:

- `m5-tiny`: lower
- `m5-small`: recommended
- `m5-balanced`: upper
- `m5-large`: upper
- `m5-xlarge`: upper

The completed fast bring-up also finished its reduced eval-rung pass on the candidate checkpoint:

| Rung | Seq len | Batch | Eval tokens | `val_bpb` | Eval seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cheap` | `2048` | `32` | `262144` | `1.282243` | `0.8` |
| `reference` | `2048` | `32` | `1572864` | `1.253053` | `4.5` |

Artifacts:

- `/home/ent/autoresearch-everywhere-sync/results/analysis/cuda_fast_ladder_fa4/report.json`
- `/home/ent/autoresearch-everywhere-sync/results/analysis/cuda_fast_ladder_fa4/report.md`
- `/home/ent/autoresearch-everywhere-sync/results/analysis/cuda_fast_ladder_fa4/promotion/platform_default.json`

The promotion bundle still marks eval calibration as not yet promotable, because fast mode ran `cheap` and `reference` but not `full`. Even so, the important platform-default result is now grounded: on GB10 with FA4, the shared CUDA ladder wants to start at `m5-small`, not `m5-tiny`.

## Why We Need A100, H100, and B200

GB10 alone is not enough.

Each of the other target systems answers a different parity question.

### A100

Purpose:

- validate Ampere/A100-family runtime behavior
- confirm the FA2-era path
- establish the oldest still-important datacenter reference class

Questions it answers:

- does the CUDA path still behave sensibly on pre-Hopper datacenter hardware?
- do trace-family mappings and Triton starter workspaces behave differently on Ampere?
- what platform default should a serious but older cloud GPU land on?

### H100

Purpose:

- validate Hopper/H100 as the closest real hardware analogue to Karpathy's upstream hand-shaped starting point
- confirm the FA3 path
- anchor the "upstream/H100 reference" story in actual repo-local measurements rather than just inherited assumptions

Questions it answers:

- what does the upstream-style preset actually look like here in this repo's CUDA trainer?
- does the bring-up machinery choose the same default it should?
- which parts of MLX's eval ladder should transfer directly to CUDA and which should not?

### B200

Purpose:

- validate datacenter Blackwell as distinct from GB10
- confirm the FA4 path on the actual large-server Blackwell target
- measure the gap between GB10-style local Blackwell and B200-style datacenter Blackwell

Questions it answers:

- what should the Blackwell datacenter reference policy be?
- how different is the platform default from GB10?
- does the CUDA kernel-lab priority structure shift materially when compute and memory headroom change that much?

## The Core Loop We Need To Reach

The parity target should be:

1. `prepare.py --engine cuda`
   - robust data/tokenizer path
   - same top-level UX as MLX

2. `train.py --engine cuda`
   - stable training
   - calibrated benchmarking
   - explicit attention-backend reporting
   - parity on the important summary fields

3. `calibrate.py --engine cuda`
   - real local search
   - candidate default selection
   - bounded zone report
   - checkpoint-backed eval calibration
   - promotion-ready output

4. runtime CUDA policy
   - trusted eval rung selection
   - explicit confidence/freshness status
   - conservative fallback behavior

5. `kernel-lab.py --engine cuda`
   - trace-first targeting
   - Triton-backed starter workspaces
   - trainer integration evidence
   - promotion-ready CUDA kernels

## Roadmap To Parity

### Phase 1: Finish CUDA trainer-loop baseline parity

Goal:
- get the CUDA engine to the same baseline operational posture as the pre-kernel-lab MLX path

Work:
- add train-vs-wall budget mode if CUDA still differs there
- align summary fields where they meaningfully map
- expose resume/checkpoint capabilities more explicitly through the shared engine boundary

Exit criteria:
- `calibrate.py --engine cuda --mode fast` can do more than one coarse fixed probe
- CUDA can mint a checkpoint at the chosen point and resume it exactly from the shared workflow

### Phase 2: Finish the eval-policy stack

Goal:
- turn the now-working checkpoint-backed rung runner into a real cheap/reference/full runtime policy story

Work:
- build CUDA `eval-batch` calibration
- build CUDA `eval-rungs`
- create seeded CUDA eval rows by hardware family
- add runtime eval-policy selection with:
  - hardware key
  - confidence
  - freshness
  - signature drift

Important note:
- CUDA does not need to copy MLX's exact implementation
- it does need comparable semantics and conservative policy behavior

Exit criteria:
- CUDA trainer reports:
  - `canonical_rung`
  - `eval_calibration_status`
  - confidence/freshness metadata
- CUDA bring-up can emit promotable eval rows

### Phase 3: Make platform bring-up truly first-class for CUDA

Goal:
- `calibrate.py --engine cuda` should feel like a real new-user flow, not a placeholder

Work:
- add CUDA local search across:
  - device batch
  - total batch
  - maybe a limited sequence/window search where appropriate
- classify lower / recommended / upper / reference zones
- produce candidate defaults by hardware family
- compare against:
  - GB10 local Blackwell reference
  - A100 datacenter reference
  - H100 upstream-style reference
  - B200 datacenter Blackwell reference

Exit criteria:
- a new CUDA machine can get a plausible candidate default from one command

### Phase 4: Connect CUDA kernel-lab to the real trainer

Goal:
- turn CUDA kernel-lab from trace-backed workspace exploration into trainer-impacting promotion

Work:
- add CUDA trainer integration hooks for narrow starter targets
- start with one or two families only
  - likely `launch_fusion`
  - likely `norm`
- add CUDA `integration-ab`
- add CUDA `integration-suite`
- make CUDA promotion conservative like MLX:
  - workspace correctness
  - microbench win
  - trace-backed relevance
  - trainer-side A/B evidence

Exit criteria:
- at least one CUDA lab target can become `integration-validated`

### Phase 5: Whole attention backends

Goal:
- move from seam-level kernels to full backend-level experiments like FlashAttention and SageAttention

This phase comes **after** the previous ones because full attention backends should be tested inside a robust trainer loop, not in isolation.

Work:
- define attention backend adapters as first-class CUDA trainer choices
- measure them across:
  - A100
  - H100
  - GB10
  - B200
- compare:
  - throughput
  - memory pressure
  - train-time overhead
  - end-to-end quality under fixed budgets

This is also where "feature parity" becomes more than operational parity. It becomes real backend competition inside the same calibrated training loop.

## Recommended Hardware Test Matrix

### Minimum viable matrix

| Hardware | Why it matters | Priority |
| --- | --- | --- |
| GB10 | real local Blackwell validation, already active | immediate |
| H100 | upstream-style reference and Hopper/FA3 validation | immediate |
| A100 | Ampere datacenter baseline and FA2-era path | near-term |
| B200 | datacenter Blackwell reference and FA4 datacenter behavior | near-term |

### Test bundles to run on each

For each machine, the first complete parity bundle should be:

1. `prepare.py --engine cuda`
2. `train.py --engine cuda --smoke`
3. `train.py --engine cuda --preset upstream --time-budget 300`
4. `calibrate.py --engine cuda --mode fast`
5. `kernel-lab.py --engine cuda capture`
6. `kernel-lab.py --engine cuda trace-profile`
7. `kernel-lab.py --engine cuda auto-review`
8. `kernel-lab.py --engine cuda deep-profile`

Then, once CUDA trainer integration exists:

9. `kernel-lab.py --engine cuda integration-suite`

## Recommendation

The repo is ready for a serious CUDA parity push now.

The order should be:

1. shared-engine trainer parity
2. eval calibration and runtime policy
3. real CUDA bring-up defaults on H100, A100, GB10, and B200
4. CUDA trainer integration for kernel-lab
5. only then push hard on whole attention backends like FlashAttention and SageAttention

That order keeps the attention-backend work inside a trustworthy training/calibration environment instead of treating it like a standalone benchmark exercise.

# Changelog

This changelog is intended to be useful for research and engineering governance, not just release bookkeeping.
Each entry records:

- `Fully human (6)`: the change was identified and authored by a human, with the agent absent or limited to review and minor revisions.
- `Human-driven (5)`: the human identified the change and specified it tightly enough that the agent mostly executed.
- `Human-directed, AI-shaped (4)`: the human set the direction or requirement, but the agent designed the concrete mechanism, structure, or validation plan.
- `AI-identified within brief, human-shaped (3)`: inside a broad human-scoped workstream, the agent surfaced the opportunity, and the human materially shaped the exact target, scope, or framing before implementation.
- `AI-identified within brief, human-approved (2)`: inside a broad human-scoped workstream, the agent surfaced the opportunity and the human approved it with little additional shaping.
- `Self-initiated, human-approved (1)`: the agent initiated the change outside explicit human direction in the thread, but still got human approval before landing it.
- `Fully autonomous (0)`: changes or experiments the agent initiated without explicit human direction or approval in the thread.
- `Grounding`: the files changed, the checks run, and any measured effects. It is intentionally unscored because it measures validation strength rather than autonomy level.

This changelog should bias toward under-claiming rather than over-claiming successful autonomy. When specific provenance attributions are corrected, prefer the more conservative tiering if there is real ambiguity. The long-term goal remains to push as much work as possible into the `Fully autonomous` category over time.
Each entry should describe not just what changed, but also the change's meaning, motivation, and intended purpose. A short nested `Meaning:`, `Motivation:`, `Purpose:` trio is a good default way to make that explicit when an entry would otherwise read like a task list.
Top-level provenance bullets are the scored units. If a point is directly derivative of a main bullet and stays at the same autonomy level, record it as a nested sub-bullet so it remains visible without changing `score`.
Each commit header should use a Linux-kernel-style subsystem prefix: `subsystem: summary`. Use the dominant subsystem rather than a file inventory. Only use a combined prefix such as `train/checkpoints:` when the change is genuinely cross-cutting and one subsystem label would be misleading. The parser treats this prefix as required so autonomy golf can be tallied by subsystem as well as by day.
Each commit entry should also show an autonomy golf score in the header. Score each provenance bullet as `Fully human = 6`, `Human-driven = 5`, `Human-directed, AI-shaped = 4`, `AI-identified within brief, human-shaped = 3`, `AI-identified within brief, human-approved = 2`, `Self-initiated, human-approved = 1`, and `Fully autonomous = 0`. `Grounding` does not contribute to the score. The header `score` is the arithmetic mean of the top-level provenance bullet weights for that entry, rounded to two decimals, so each commit stays on a bounded `0..6` spectrum.
Preserve the summed provenance surface separately as `complexity`. Compute it as the sum of top-level provenance weights, plus `+1` for each nested sub-bullet under provenance items scored `3` or higher, excluding `Meaning:`, `Motivation:`, and `Purpose:` narrative lines. When `complexity` is numerically identical to the bounded `score`, omit it from the header as redundant; the parser treats omission as an implicit equality.
Entries should omit empty provenance sections rather than spelling out `None in this entry.`
If an entry has no measurements yet, it should say so explicitly.
When provenance is ambiguous, prefer `Human-directed, AI-shaped` over `AI-identified within brief, human-shaped`, prefer `AI-identified within brief, human-shaped` over `AI-identified within brief, human-approved`, prefer `AI-identified within brief, human-approved` over `Self-initiated, human-approved`, and prefer `Self-initiated, human-approved` over `Fully autonomous`.
This branch does not admit fully human-authored code changes. If a change must be authored entirely by a human, it belongs in a fork rather than this branch's mainline history.
Grounding should also be conservative. When a claim is about performance, stability, or behavioral improvement, prefer the strongest practical evidence over the quickest smoke pass, and record the actual strength of that evidence rather than the intended standard.
For benchmarked changes, "strong enough" means long enough and heavy enough to produce a high-signal result on the changed behavior. Choose a run shape where the affected path executes enough times to matter. For example, checkpoint-overhead claims should usually be grounded with a run that produces many checkpoint saves rather than only one or two, and scaling claims should prefer a model/preset large enough for the bottleneck to show up clearly.
For checkpoint semantic changes, save/restore cost is not enough by itself. Prefer a convergence benchmark that compares uninterrupted training, exact midpoint resume, and approximate midpoint resume under the same total optimizer-step budget, then records the end-state differences in loss, validation BPB, and parameter drift.
On this hardware, the default canonical matched benchmark window for optimization grounding is `60s`, not `30s`. Use shorter runs for smoke checks or when the changed path cannot practically support a longer benchmark, and say so explicitly when you do.

## Latest

### New commit — train/lr: add staged unembedding multiplier discovery — score `4` — complexity `6`

**Human-directed, AI-shaped (4)**

- Expose `unembedding_lr_multiplier` as the next staged LR-discovery lever after `matrix`, so the shared sweep can tune the `lm_head` / unembedding group without another trainer or engine surface change.
  - Meaning: staged discovery is no longer limited to `global -> matrix`; the same shared sweep can now continue into `global -> matrix -> unembedding`, carrying forward the previously discovered multipliers and ranking the new stage on the unembedding axis itself.
  - Motivation: the shared LR-profile surface already had a distinct unembedding/head group, and the staged matrix work proved the discovery plumbing was generic enough to support another lever cheaply. The next useful question after matrix is whether the head/unembedding group wants to move off the inherited anchor once global and matrix have already been adjusted.
  - Purpose: widen grouped LR discovery one step further while keeping the platform boundary stable, so future grouped searches can keep adding levers instead of reopening trainer-specific LR seams.
  - Extend `autoresearch_platform.lr_discovery` with a first-class `unembedding` lever mapped onto `unembedding_lr_multiplier`, so staged runs can carry the existing probe cache, near-tie reporting, and stage-local summaries straight into a head/unembedding sweep.
  - Extend `tools/discover_lr.py` help and lever selection to advertise `unembedding` alongside `global` and `matrix`, then validate the new stage with a real MLX `m5-tiny` run and a staged summary plot.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_platform/lr_discovery.py`
  - `tools/discover_lr.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/lr_discovery.py tools/discover_lr.py discover_lr.py`
  - `python3 discover_lr.py --help | rg -n "discovery-levers|unembedding|matrix"`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python discover_lr.py --engine mlx --preset m5-tiny --time-budget 1.5 --discovery-levers global matrix unembedding --streaming-eval-interval-steps 1 --streaming-eval-tokens 16384 --output-dir results/analysis/lr_discovery_mlx_m5tiny_global_matrix_unembedding_test`
    - Real staged MLX result: `global=1.8340081`, `matrix=0.54525387`, `unembedding=1.0905077`, with stage-local near-tie reporting still working and the final longer-horizon recommendation falling back to `unembedding=1.0`.
  - `UV_CACHE_DIR=/tmp/uv-cache uv run --with matplotlib python - <<'PY' ...`
    - Wrote staged summary plot: `results/analysis/lr_discovery_mlx_m5tiny_global_matrix_unembedding_test/staged_lr_discovery_plot.png`.

### New commit — train/lr: add optional bowl-finding discovery mode — score `4` — complexity `8`

**Human-directed, AI-shaped (4)**

- Add an opt-in `find-bowl` discovery mode that keeps extending the sweep until the current winner is bracketed by meaningfully worse probes on both sides, or the extra-probe budget is exhausted.
  - Meaning: the shared LR sweep can now distinguish between "best at this budget" and "actually bracketed a local bowl," recording explicit bowl status, side-gap fractions, and budget-exhausted outcomes per stage instead of pretending every sweep ends with the same level of certainty.
  - Motivation: the staged `global -> matrix` work exposed that near-ties and shallow edges are not the same thing as a real bracketed optimum. For longer or more consequential runs, we want an optional stronger stop condition than the current bounded opportunistic refinement.
  - Purpose: let discovery stay cheap by default while adding a more trustworthy mode for longer-horizon or grouped-lever searches where proving both-side degradation matters more than minimizing probe count.
  - Extend `autoresearch_platform.lr_discovery` with `standard` vs `find-bowl` sweep modes, configurable bowl degradation thresholds and extra-probe caps, and stage-level bowl reporting that prefers the closest qualifying degraded probe on each side while still exposing the nearest-side gaps for context.
  - Add a stage-anchor pathology guard to discovery probes: if a jump or follow-on probe exceeds `3x` the anchor-established online AUC/BPB baseline, the trainer stops that probe early, discovery marks it pathological, excludes it from ranked evidence, and falls back to ordinary doubling in that direction instead of letting absurd jump probes pollute the bowl search.
  - Extend `tools/discover_lr.py` with `--discovery-mode`, `--bowl-auc-fraction`, and `--bowl-max-extra-probes`, plus CLI/Markdown summary output for bowl status alongside the existing near-tie reporting.
  - Validate the mode with both synthetic sweeps and real MLX `m5-tiny` probe runs; the updated staged run confirmed that one pathological matrix jump is now discarded, the stage falls back to finite doublings, and the bowl still closes cleanly around `matrix=2`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_cuda/train.py`
  - `autoresearch_mlx/train.py`
  - `autoresearch_platform/cuda_engine.py`
  - `autoresearch_platform/engines.py`
  - `autoresearch_platform/lr_discovery.py`
  - `autoresearch_platform/mlx_engine.py`
  - `tools/discover_lr.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/lr_discovery.py tools/discover_lr.py discover_lr.py autoresearch_platform/engines.py autoresearch_platform/mlx_engine.py autoresearch_platform/cuda_engine.py autoresearch_mlx/train.py autoresearch_cuda/train.py`
  - `python3 - <<'PY' ...` synthetic bowl-confirmed smoke with `jump_threshold=999`, showing one outward extension was enough to confirm a bowl
  - `python3 - <<'PY' ...` synthetic budget-exhausted smoke with `jump_threshold=999`, showing the new mode exits honestly as `budget-exhausted-no-bowl` when both-side degradation never becomes large enough
  - `python3 - <<'PY' ...` synthetic pathological-jump fallback smoke, showing a pathological extrapolated jump is discarded, the direction falls back to ordinary doubling, and the finite continuation probe still ranks normally
  - `python3 - <<'PY' ...` history round-trip smoke confirming trainer-emitted `probe_pathology_triggered` / `probe_pathology_reason` propagate back into `AdaptiveLrProbe`
  - `python3 discover_lr.py --help | rg -n "discovery-mode|bowl-auc-fraction|bowl-max-extra-probes"`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python discover_lr.py --engine mlx --preset m5-tiny --time-budget 1.5 --discovery-mode find-bowl --discovery-levers global --streaming-eval-interval-steps 1 --streaming-eval-tokens 16384 --output-dir results/analysis/lr_discovery_mlx_m5tiny_find_bowl_test`
    - Real MLX result: `global=1.8340081`, `bowl-confirmed`, `left_gap_fraction=0.05799`, `right_gap_fraction=0.08149`, `nearest_right_gap_fraction=0.00291`, `extra_probes_used=0`.
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python discover_lr.py --engine mlx --preset m5-tiny --time-budget 1.5 --discovery-mode find-bowl --discovery-levers global matrix --streaming-eval-interval-steps 1 --streaming-eval-tokens 16384 --output-dir results/analysis/lr_discovery_mlx_m5tiny_global_matrix_find_bowl_test3`
    - Real staged MLX result: `global=1.8340081`, `matrix=2`, matrix-stage `discarded_pathological_probes=1`, and the ranked finite matrix probes remained `2, 4, 1, 0.5, 8, 0.25`.

### New commit — train/lr: add staged matrix multiplier discovery — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Extend LR discovery from a scalar-only probe loop into a staged shared surface that can discover `matrix_lr_multiplier` after the global scalar without re-opening another backend-specific trainer seam.
  - Meaning: `discover_lr.py` can now run ordered discovery levers, keeping the existing global `lr_multiplier` sweep as stage one and then optionally running the same adaptive sweep over `matrix_lr_multiplier` while holding the discovered global multiplier fixed.
  - Motivation: the new shared LR-profile surface made grouped tuning possible, but discovery was still trapped on one scalar axis. The first high-value extension is `matrix` vs rest, because it exercises the shared grouped surface without exploding the search into an ungrounded multi-dimensional probe grid.
  - Purpose: let discovery explore the first non-scalar LR lever through the same platform boundary both backends already use, so future grouped discovery can add more axes without rewriting the tool or trainer interfaces again.
  - Add staged lever support to `autoresearch_platform.lr_discovery`, including ordered `global -> matrix` sweeps, full-multiplier probe caching across stages, and stage-local probe metadata so matrix-stage ranking is driven by the matrix value being searched rather than by the inherited global scalar.
  - Extend `tools/discover_lr.py` with `--discovery-levers`, grouped-multiplier probe launching, staged JSON/Markdown summaries, and final recommended train args that include both the discovered global and matrix multipliers when present.
  - Harden the shared LR-profile parsing helpers so both mappings and argparse/dataclass-style objects can feed the new grouped surface; the first real MLX staged sweep flushed out that gap.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_platform/lr_discovery.py`
  - `autoresearch_platform/lr_profile.py`
  - `tools/discover_lr.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/lr_profile.py autoresearch_platform/lr_discovery.py tools/discover_lr.py discover_lr.py autoresearch_mlx/train.py autoresearch_cuda/train.py autoresearch_mlx/optim.py autoresearch_cuda/config.py`
  - `python3 - <<'PY' ...` synthetic staged-sweep smoke confirming `global -> matrix` discovery and full-multiplier cache reuse across stages
  - `python3 - <<'PY' ...` namespace-parsing smoke confirming `lr_profile_from_mapping()` and `lr_multipliers_from_mapping()` accept argparse/dataclass-style objects as well as plain mappings
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python discover_lr.py --engine mlx --preset m5-tiny --time-budget 1.5 --discovery-levers global matrix --streaming-eval-interval-steps 1 --streaming-eval-tokens 16384 --output-dir results/analysis/lr_discovery_mlx_m5tiny_global_matrix_test`
    - Real staged MLX result: `global=1.8340081`, `matrix=2`, `13` logical probes with one cached cross-stage reuse, and stage-local near-tie reporting for both the global and matrix sweeps.

### New commit — train/lr: add a shared grouped LR profile surface — score `4` — complexity `9`

**Human-directed, AI-shaped (4)**

- Lift LR handling onto a shared grouped-profile surface so MLX and CUDA consume the same base preset rates, per-group multipliers, and resolved optimizer-group rates instead of each trainer carrying its own partial LR model.
  - Meaning: the repo now has an explicit shared LR profile layer in `autoresearch_platform.lr_profile` with compact preset-owned rates (`embedding`, `unembedding`, `matrix`, `scalar`), optional grouped multipliers on top of the existing global `lr_multiplier`, and one shared resolver that expands that compact profile into the actual optimizer groups both backends train with (`lm_head`, `embedding`, `value_embedding`, `resid`, `x0`, `matrix`).
  - Motivation: the discovery work exposed that the current LR surface was inconsistent and incomplete. CUDA presets already owned grouped LR ratios, MLX still hard-coded them in the trainer, and both trainers collapsed the public surface back to a single scalar `lr_multiplier`, which would make any future multi-axis discovery or manual grouped tuning brittle and backend-specific.
  - Purpose: create one stable LR-management boundary that can support scalar discovery today, richer grouped tuning later, and resume-safe persistence of the actual LR surface instead of relying on whatever the current preset code happens to say at restore time.
  - Add `autoresearch_platform.lr_profile` as the shared home for base preset LR profiles, grouped multipliers, resolved per-optimizer-group LR expansion, and backward-compatible parsing from old scalar-only checkpoint metadata.
  - Move both preset surfaces onto that shared profile: MLX presets now carry `lr_profile` instead of relying on trainer globals, and CUDA presets now expose the same grouped LR profile through `CudaRunPreset`.
  - Extend both trainer CLIs, run-config payloads, summaries, and resume guards to carry `embedding`, `unembedding`, `matrix`, and `scalar` LR multipliers alongside the existing global multiplier, while preserving compatibility with old checkpoints that only stored `lr_multiplier`.
  - Refactor both optimizer paths to consume the same resolved LR profile up front and then apply only the time-schedule factor during training, instead of mixing preset rates, dmodel scaling, and the global multiplier differently inside each backend.
  - Expose the shared LR profile on the engine preset catalog and on engine probe launch paths so future grouped discovery work can flow through the platform boundary without another trainer-specific surface rewrite.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_platform/lr_profile.py`
  - `autoresearch_platform/__init__.py`
  - `autoresearch_platform/engines.py`
  - `autoresearch_platform/mlx_engine.py`
  - `autoresearch_platform/cuda_engine.py`
  - `autoresearch_mlx/optim.py`
  - `autoresearch_mlx/train.py`
  - `autoresearch_cuda/config.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/lr_profile.py autoresearch_platform/__init__.py autoresearch_platform/engines.py autoresearch_platform/mlx_engine.py autoresearch_platform/cuda_engine.py autoresearch_mlx/optim.py autoresearch_mlx/train.py autoresearch_cuda/config.py autoresearch_cuda/train.py`
  - `python3 - <<'PY' ...` shared LR-profile smoke covering grouped-multiplier application plus resolved per-group rates at a non-reference model width
  - `python3 - <<'PY' ...` backward-compatibility smoke confirming old scalar-only run-config payloads still restore as grouped multipliers with `embedding/unembedding/matrix/scalar = 1.0`

### New commit — train/lr: add adaptive multiplier discovery and near-tie reporting — score `4` — complexity `14`

**Human-directed, AI-shaped (4)**

- Add a dedicated adaptive LR-multiplier sweep that runs real short training probes against deterministic streaming validation instead of relying on fixed preset ratios.
  - Meaning: `discover_lr.py` and `tools/discover_lr.py` now run same-engine MLX or CUDA probes, keep the preset's per-group LR ratios intact, and search only the scalar `lr_multiplier` that scales those ratios together.
  - Motivation: once the trainers exposed deterministic streaming AUC and cycle-complete honest metrics, LR choice no longer needed to be hard-coded or tuned manually one long run at a time; it could be discovered from a small budget of directly comparable probes.
  - Purpose: make LR selection a reusable front-door workflow that can pick a strong multiplier before longer training runs without forcing each experiment to build its own probe harness.
  - Add `autoresearch_platform.lr_discovery` with adaptive anchor/up/down probing, log-space duplicate suppression, directional refinement, and backend-agnostic sweep orchestration around a shared `run_probe()` contract.
  - Add `tools/discover_lr.py` plus the top-level `discover_lr.py` wrapper so the sweep can launch real trainer probes, persist per-probe histories and logs, and write ranked JSON/Markdown summaries with ready-to-run follow-up train args.
  - Rank probes by streaming-validation AUC while recording supporting tail metrics (`min_bpb`, `last20_bpb`, `honest_bpb`, completed cycles) so the sweep result preserves both short-run learning speed and end-of-probe quality context.
- Refine the LR search policy so it stays coarse-first, zooms in only when the winner has enough edge to justify it, and reports close finishes honestly instead of overclaiming one precise multiplier.
  - Meaning: the sweep now completes the initial `anchor`, `2x`, and `0.5x` bracket before any local zoom, dedupes distinct multipliers when deciding whether to refine, and emits an explicit `near_tie` result when the top probes are effectively flat at the chosen probe budget.
  - Motivation: real 5090 sweeps exposed two failure modes in the first pass: duplicate coarse multipliers could suppress the true refine trigger, and tiny AUC gaps were being reported as decisive winners even when the real takeaway was "keep this as the default pick, but the top pair are effectively tied."
  - Purpose: keep the probe budget efficient, avoid spurious reruns, and make the sweep output reflect actual certainty instead of just the current sort order.
  - Restrict fine refinement to a post-coarse local search around the incumbent, using a tighter duplicate tolerance and shrinking multiplicative steps instead of interleaving local probes into the coarse bracket.
  - Add near-tie reporting to the shared sweep result plus CLI/Markdown output, including the lower-LR longer-horizon alternative when it is distinct and clearer wording when the winner is already the lower-LR member of the top pair.
  - Fix duplicate handling in refinement and near-tie decisions so repeated coarse points do not mask the real distinct runner-up or prevent refinement around the actual winner.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_platform/lr_discovery.py`
  - `tools/discover_lr.py`
  - `discover_lr.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/lr_discovery.py tools/discover_lr.py discover_lr.py`
  - `python3 discover_lr.py --help`
  - `python3 - <<'PY' ...` synthetic sweep smoke covering a clear coarse winner that triggers local refinement, a near-tied top pair that stops without extra zoom, and a duplicate-coarse case where the true distinct runner-up still triggers refinement around the real winner.
- Measurements:
  - On the RTX 5090 `m5-tiny` CUDA sweep, the refined policy evaluated `7` probes and moved the short-run pick from the anchor `1.0x` to `0.91700404x`, improving probe AUC from `1.518842` to `1.518370` (about `0.03%`) and improving a follow-up `10`-reference-cycle explicit end-state reference eval from `1.389380` to `1.389000`.
  - On the RTX 5090 `m5-large` CUDA sweep with FA4 working through the default `uv` front door, the corrected coarse-first policy evaluated `5` probes and returned `0.91700404x` as the short-run pick while keeping `1.0x` and `1.0905077x` inside the reported near-tie band rather than overstating a precise single optimum.

### New commit — train/eval: add shared streaming honest eval hierarchy — score `4` — complexity `14`

**Human-directed, AI-shaped (4)**

- Add a shared deterministic streaming-validation surface to both trainers so short runs can produce directly comparable AUC and cycle-complete honest metrics without paying for standalone eval passes.
  - Meaning: both MLX and CUDA now expose the same online-eval contract: one deterministic validation batch every `N` steps, rolling cycle summaries that emit `honest_val_bpb` when the configured slice completes, optional history JSON, and an opt-in rule to finish the current cycle before stopping so the final online metric is not truncated.
  - Motivation: the new probe workflow needed two capabilities the trainers did not expose: identical validation ordering across short runs, and a cheap amortized signal stable enough to score the ramp before a full canonical eval would be worth pausing for.
  - Purpose: make early-training online eval a first-class shared surface that downstream probes and longer runs can both rely on without splitting MLX and CUDA behavior.
  - Add shared `autoresearch_platform.streaming_eval` types and engine/result-surface fields for streaming AUC, honest BPB, cycle counts, history output, and derived reference reporting.
  - Wire both trainers and the shared engine surface to accept `--streaming-eval-*` plus `--lr-multiplier`, emit `streaming_val_auc` / `honest_val_bpb`, and persist optional point-by-point streaming history while keeping the multiplier as a scale factor over the preset LR ratios.
  - Refine the online-eval hierarchy so repeated fixed-sample `cheap` cycles remain available for fast comparable probes, while `subref-one-sixth` walks disjoint one-sixth shards of `reference` and derives `honest_reference_bpb` every six completed subcycles.
- Close the CUDA front door and Blackwell bring-up gaps that the new online-eval surface exposed.
  - Meaning: `uv run train.py --engine cuda --smoke` now works as a true tiny end-to-end sanity pass, the CUDA batch-eval helper now runs under autocast instead of dying on bf16/float mismatches, and the repo’s default `uv` env now uses Python `3.12` so the published RTX 5090 FA4 wheel can load through the advertised front door.
  - Motivation: first contact on the 5090 Runpod box exposed three real parity gaps: CUDA had no usable `--smoke` path, the new online eval crashed outside an autocast region, and the `3.10` repo pin prevented the published `torch 2.9 / cu12` FlashAttention wheel from being used in the default env.
  - Purpose: make the shared streaming probe surface operational on the current Blackwell reference path instead of leaving the CUDA side only theoretically wired.
  - Add MLX-style `--smoke` handling to `autoresearch_cuda.train`, including tiny safe overrides and a smoke eval rung.
  - Wrap CUDA batch BPB eval in autocast inside `autoresearch_cuda.prepare` so both standalone and streaming batch eval share the same bf16-safe behavior.
  - Pin `.python-version` to `3.12` and update the README fast-track wording accordingly.

**Grounding**

- Files:
  - `.python-version`
  - `CHANGELOG.md`
  - `autoresearch_platform/engines.py`
  - `autoresearch_platform/streaming_eval.py`
  - `autoresearch_platform/cuda_engine.py`
  - `autoresearch_platform/mlx_engine.py`
  - `autoresearch_cuda/eval_policy.py`
  - `autoresearch_cuda/prepare.py`
  - `autoresearch_cuda/train.py`
  - `autoresearch_mlx/data.py`
  - `autoresearch_mlx/eval_policy.py`
  - `autoresearch_mlx/train.py`
  - `README.md`
- Validation:
  - `python3 -m py_compile autoresearch_platform/engines.py autoresearch_platform/mlx_engine.py autoresearch_platform/cuda_engine.py autoresearch_platform/streaming_eval.py autoresearch_mlx/data.py autoresearch_mlx/eval_policy.py autoresearch_mlx/train.py autoresearch_cuda/prepare.py autoresearch_cuda/train.py`
  - `uv run python train.py --engine cuda --help | rg -n -- "--smoke|benchmark-skip-eval|no-compile|streaming-eval-interval-steps"`
  - `uv run python -V` after updating `.python-version` to `3.12`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -m autoresearch_mlx.train --smoke --time-budget 0.2 --benchmark-skip-eval --no-checkpoint --streaming-eval-interval-steps 1 --streaming-eval-tokens 1024 --complete-streaming-eval-cycle --streaming-eval-history-output /tmp/autoresearch_mlx_streaming_smoke/history.json`
  - `uv run python - <<'PY' ...` structural check that `cheap` stays a repeated subset while six disjoint `subref-one-sixth` shards exactly reconstruct `reference`
  - `uv run python -m autoresearch_mlx.train --preset m5-tiny --token-budget 4718592 --streaming-eval-interval-steps 1 --streaming-eval-tokens 262144 --streaming-eval-history-output /tmp/autoresearch_subref_mlx/history.json --benchmark-skip-eval --no-checkpoint`
  - real RTX 5090 Runpod checks after syncing the patched CUDA trainer files into the Runpod checkout:
    - `uv run prepare.py --engine cuda`
    - `uv run train.py --engine cuda --smoke`
    - `uv run python -m autoresearch_cuda.train --smoke --token-budget 32768 --streaming-eval-interval-steps 1 --benchmark-skip-eval --no-checkpoint --no-compile`
    - `uv run python -m autoresearch_cuda.train --smoke --token-budget 32768 --streaming-eval-interval-steps 1 --streaming-eval-mode cheap --streaming-eval-tokens 262144 --benchmark-skip-eval --no-checkpoint --no-compile`
    - `uv run python -m autoresearch_cuda.train --smoke --token-budget 32768 --streaming-eval-interval-steps 1 --streaming-eval-mode both --streaming-eval-tokens 262144 --benchmark-skip-eval --no-checkpoint --no-compile`
    - `python3 - <<'PY' ...` summary readback of `/tmp/autoresearch-cuda-subref-smoke.json`, `/tmp/autoresearch-cuda-cheap-smoke.json`, and `/tmp/autoresearch-cuda-both-smoke.json`
- Measurements:
  - local MLX streaming-eval smoke on the trainer path completed `12` deterministic eval points over `6` full cycles, ending with `streaming_val_auc=2.821716` and `honest_val_bpb=2.650096` while still skipping the expensive final canonical eval.
  - the new MLX canonical-plan relationship is now explicit rather than accidental: with the current eval contracts, `cheap` is `64` batches, `subref-one-sixth` is also `64`, and `reference` is `384`; the six `subref-one-sixth` chunks are pairwise disjoint, their union exactly matches the full `reference` plan, and the repeated `cheap` plan stays a fixed subset of the same reference horizon rather than becoming the first sixth verbatim.
  - a real MLX trainer run on `m5-tiny` with one streamed eval batch every training step and a `4,718,592`-token budget completed `384` training steps, `384` repeated-cheap points, `384` `subref-one-sixth` points, `6` cheap cycles, `6` `subref-one-sixth` cycles, and `1` derived `reference` cycle, ending with `honest_val_bpb=1.790825`, `honest_subref_one_sixth_bpb=1.793900`, `honest_reference_bpb=1.951716`, `streaming_val_auc=1.944725`, `training_seconds=52.9`, and `streaming_eval_total_seconds=15.4`.
  - a real RTX 5090 / Runpod smoke now completes on the patched front door using the exact README command, reporting `hardware_key=nvidia-blackwell-rtx50-31gb`, `canonical_rung=smoke`, `eval_calibration_status=smoke`, `val_bpb=3.144333`, `training_seconds=1.0`, `peak_vram_mb=142.1`, and `resolved_attention_backend=torch-sdpa`.
  - the first real CUDA streaming smoke immediately exposed and grounded a real parity bug in the new online-eval path: `evaluate_bpb_batch_stats(...)` was calling the model outside the CUDA autocast context and died on `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float`; the fix was to make the helper enter `torch.amp.autocast(...)` internally so both standalone eval and online eval are safe when called outside a surrounding autocast block.
  - after that fix, the same RTX 5090 host completed all three CUDA streaming modes on the tiny smoke shape with `--token-budget 32768` and one eval tick per training step:
    - default `subref-one-sixth`: `64` subref points, `16` completed subref cycles, `2` derived reference cycles, `honest_subref_one_sixth_bpb=2.495688`, `honest_reference_bpb=2.569155`, `training_seconds=0.5`, `total_seconds=3.3`
    - `cheap`: `64` repeated-cheap points, `16` cheap cycles, `streaming_val_auc=2.660665`, `honest_val_bpb=2.538248`, `training_seconds=0.5`, `total_seconds=4.0`
    - `both`: `64` repeated-cheap points, `64` subref points, `16` cheap cycles, `16` subref cycles, `2` derived reference cycles, `streaming_val_auc=2.660214`, `honest_val_bpb=2.537253`, `honest_subref_one_sixth_bpb=2.495301`, `honest_reference_bpb=2.568650`, `training_seconds=0.5`, `total_seconds=6.1`
  - that same host also grounded the bug this patch closes: before the smoke flag was added, the committed CUDA front door rejected `--smoke`, and a minimal fallback `upstream` probe on the same machine OOMed immediately under the normal default shape.
  - on the RTX 5090 host, the repo’s old `.python-version=3.10` pin prevented FA4 from coming up through the normal `uv` path because upstream only published the `torch 2.9 / cu12` Linux wheel for `cp312`; after rebuilding a `3.12` env and installing `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`, `train.py --engine cuda --smoke` resolved `installed:flash_attn.flash_attn_interface` instead of `torch-sdpa`.

## Committed History

### March 20, 2026 — `72fe88e` — platform/checkpoints: unify exact checkpoint policy and telemetry across MLX and CUDA — score `3` — complexity `12`

**AI-identified within brief, human-shaped (3)**

- Turn exact checkpointing into a shared cross-backend surface: async exact writes, one auto-cadence planner, GB10-grounded CUDA save-cost calibrations, token-aware checkpoint policy, and a common phase-timing summary contract.
  - Meaning: MLX and CUDA now speak one checkpoint-policy language. Both backends support exact resumability with shared interval parsing and auto-planning, CUDA now has a real async exact path instead of a dead-end approximate branch, and both trainers can emit a common timing summary that downstream tools can consume without backend-specific scraping.
  - Motivation: the checkpoint stack had split into too many partially overlapping concepts: MLX-only async plumbing, CUDA-only exact semantics, fixed CUDA auto intervals, token-budget special cases, and summary telemetry that diverged enough to make cross-backend reasoning brittle. The right fix was not one more backend-specific patch, but a unified checkpoint and telemetry surface grounded in real hardware measurements.
  - Purpose: make resumability and checkpoint overhead policy trustworthy across the two active engines, keep the default 300-second path overhead-free, and give calibration/profiling/reporting code one stable interface for checkpoint behavior and step timing.
  - Add `autoresearch_platform/async_checkpoint.py` as the shared home for background checkpoint write orchestration, then refactor MLX to use that shared writer while keeping MLX-specific snapshot capture and host-array serialization in place.
  - Extend `autoresearch_cuda.checkpoints` with exact CPU snapshot capture, async snapshot writing, `sync|async` save-mode metadata, and a reusable CUDA async-writer factory.
  - Wire `autoresearch_cuda.train` through a real `--checkpoint-save-mode {sync,async}` surface, preserve the chosen save mode across resume, include it in auto-checkpoint path naming, and make the trainer queue or flush checkpoints correctly depending on save mode.
  - Make the shared checkpoint interval chooser a single auto-checkpoint plan surface in `autoresearch_platform.checkpoint_policy`, so both backends now ask one shared helper for time-budget and token-budget auto cadence while deferring the first automatic checkpoint until elapsed training time is strictly `>300s` and keeping the token-budget default at the earlier of `300s` or `50Mtok` after that point.
  - Seed MLX with calibrated `exact/sync` and `exact/async` rows from the historical ABAB benchmarks, then replace CUDA's fallback-only path with real GB10 `exact/sync` and `exact/async` calibration rows for both the balanced and xlarge parameter bands.
  - Extend the shared chooser so token-budget auto-checkpointing now uses the same measured save-cost recommendation as time-budget mode, expressed as an earlier-of hybrid interval such as `10m or 50M tok`, instead of bypassing the calibration table and hard-coding `300s,50Mtok`.
  - Add CUDA-side checkpoint timing telemetry to the trainer summary (`checkpoint_percent`, `checkpoint_write_percent`, checkpoint counts, cumulative checkpoint seconds) plus a new `tools/profile_cuda_checkpoint_overhead.py` harness, then use that harness on GB10 to ground the CUDA calibration rows in repeated-save measurements rather than synthetic estimates.
  - Refactor per-step timing onto a shared platform telemetry surface so MLX and CUDA can now emit a common timing summary (`train_tflops`, `compute_share_percent`, `phase_*_percent`, `other_step_percent`, shared steady/all-step window labels) plus a shared `input_pipeline_percent` observation for host-side batch staging, while still preserving backend-specific extras such as CUDA peak-FLOPS utilization and avoiding the mistake of treating overlapped CUDA input work as an additive wall-time phase.
  - Add `docs/telemetry-architecture.md` as the current primer/reference for the telemetry stack, covering the architecture map, additive-vs-overlapping timing semantics, artifact flow, the MLX/CUDA gaps matrix, and a glossary of the active terms that calibration and profiling code now depend on.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_platform/async_checkpoint.py`
  - `autoresearch_platform/checkpoint_policy.py`
  - `docs/telemetry-architecture.md`
  - `autoresearch_mlx/checkpoint_policy.py`
  - `autoresearch_mlx/checkpoints.py`
  - `autoresearch_mlx/train.py`
  - `autoresearch_cuda/checkpoint_policy.py`
  - `autoresearch_cuda/checkpoints.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/checkpoint_policy.py autoresearch_platform/async_checkpoint.py autoresearch_mlx/checkpoint_policy.py autoresearch_mlx/checkpoints.py autoresearch_mlx/train.py autoresearch_cuda/checkpoint_policy.py autoresearch_cuda/checkpoints.py autoresearch_cuda/train.py`
  - `python3 -m py_compile autoresearch_platform/summary.py tools/profile_cuda_resume_convergence.py tools/profile_cuda_checkpoint_overhead.py`
  - `uv run --with torch python - <<'PY' ...` CPU smoke covering sync and async CUDA checkpoint writes, metadata/save-mode round-tripping, and exact loadback of async-written model weights
  - `uv run --with torch python -m autoresearch_cuda.train --help | rg -n "checkpoint-save-mode|checkpoint-interval|no-checkpoint"`
  - `python3 - <<'PY' ...` shared due-at-threshold smoke confirming automatic intervals do not fire at exactly `300.0s`, do fire once elapsed training time is `>300s`, and still let explicit intervals fire earlier when requested
  - `uv run python - <<'PY' ...` MLX chooser smoke confirming the shared auto-checkpoint plan now respects the strict `>300s` first-checkpoint gate in both time-budget and token-budget mode (`exact/sync -> 5m`, `exact/async -> 5m`, token mode -> `5m or 50M tok`)
  - `uv run --with torch python - <<'PY' ...` CUDA chooser smoke confirming the GB10-grounded rows produce calibrated plans instead of fallback defaults: balanced-band `exact/sync -> 10m`, balanced-band `exact/async -> 5m`, xlarge-band `exact/sync -> 10m`, xlarge-band `exact/async -> 15m`, with token-budget mode reusing those measured intervals as earlier-of hybrids with `50M tok`
  - `uv run python -m autoresearch_mlx.train --smoke --time-budget 1 --benchmark-skip-eval --no-checkpoint`
  - `uv run --with torch python tools/profile_cuda_checkpoint_overhead.py --help`
  - `uv run --with torch python tools/profile_cuda_resume_convergence.py --help`
- Measurements:
  - local CPU smoke wrote a valid async CUDA checkpoint bundle with `capture_seconds=0.000360`, `async_write_seconds=0.001458`, and `completed_count=1`, then loaded the async-written model weights back exactly.
  - the shared chooser now keeps both MLX save modes above the new time-budget floor while still reflecting their distinct calibrated blocking costs: for the `m5-large` parameter band, both `exact/sync` and `exact/async` land on the shared `5m` floor, with measured save-only overhead fractions of `0.0233%` and `0.0100%` respectively.
  - GB10 repeated-save overhead profiling now grounds CUDA's balanced band (`m5-balanced`, `seq=1024`, `db=32`, `tb=32768`) at `0.5216s/save` for `exact/sync` and `0.2970s/save` for `exact/async`, which yields calibrated auto plans of `10m` and `5m` respectively under the shared `0.1%` save-only overhead cap.
  - The same GB10 harness grounds CUDA's xlarge band (`m5-xlarge`, `seq=2048`, `db=16`, `tb=32768`) at `0.3804s/save` for `exact/sync` and `0.6936s/save` for `exact/async`, which yields calibrated auto plans of `10m` and `15m`; on this heavier band the async path is more blocking than sync because snapshot capture dominates.
  - A local MLX smoke on the new shared telemetry path still completed cleanly and emitted both the legacy aliases and the new shared fields in the final summary, including `compute_share_percent=99.80`, `input_pipeline_percent=0.15`, `phase_loader_percent=0.15`, `phase_grad_percent=66.95`, `phase_accumulate_percent=7.55`, and `phase_optimizer_percent=25.30`.

### March 14, 2026 — `3e64c13` — platform/checkpoints: add token-budget checkpoint parity to the MLX trainer — score `3` — complexity `7`

**AI-identified within brief, human-shaped (3)**

- Finish the next checkpoint-parity step by giving MLX the same token-budget and token-aware checkpoint behavior that CUDA already has.
  - Meaning: MLX can now run against a fixed token target, auto-enable checkpoints once that target is long enough to warrant resumability, and accept checkpoint cadence in the same shared forms as CUDA: time-only (`300s`), token-only (`50Mtok`), or hybrid earlier-of (`300s,50Mtok`).
  - Motivation: after landing the shared checkpoint-policy surface, MLX still had two real parity gaps: it remained a wall-time-only trainer, and its checkpoint trigger still stored cadence as a bare float of seconds.
  - Purpose: remove the last obvious budget-shape mismatch between MLX and CUDA, make token-denominated checkpoint cadence a first-class cross-backend feature, and ensure long fixed-token MLX runs get the same safe-by-default resumability as long CUDA runs.
  - Extend `autoresearch_mlx/train.py` with `--token-budget`, token-budget stopping logic, token-aware progress/remaining reporting, and token-budget telemetry fields so MLX can now run fixed-token experiments instead of only fixed-second ones.
  - Add the CUDA-style automatic checkpoint enablement on the MLX side for token-budgeted runs once `token_budget >= 50_000_000`, using the shared earlier-of `300s or 50M tok` default interval.
  - Update the MLX trainer to store the canonical checkpoint interval spec string, parse and normalize explicit intervals on input and resume, and use the shared `checkpoint_interval_due(...)` logic instead of a hard-coded elapsed-seconds comparison.
  - Re-export the shared interval parsing and formatting helpers from `autoresearch_mlx.checkpoint_policy` so the MLX surface stays symmetrical with CUDA while keeping MLX-specific calibration tables and path-building local.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_mlx/checkpoint_policy.py`
  - `autoresearch_mlx/eval_telemetry.py`
  - `autoresearch_mlx/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/checkpoint_policy.py autoresearch_platform/checkpoint_policy.py`
  - `python3 - <<'PY' ...` shared-policy smoke covering hybrid interval parsing/formatting and due-at-threshold behavior (`300s,50Mtok`, `50Mtok`)
  - `uv run python -m autoresearch_mlx.train --smoke --token-budget 4096 --checkpoint-interval 1024tok --checkpoint-path /tmp/autoresearch-mlx-token-smoke-XXXXXX/checkpoint --benchmark-skip-eval`
  - `uv run python -m autoresearch_mlx.train --resume-from /tmp/autoresearch-mlx-token-smoke-XXXXXX/checkpoint --token-budget 8192`
- Measurements:
  - local MLX smoke run saved exact checkpoints repeatedly on the requested `1024tok` cadence and resumed cleanly from `step=8` to `step=16` under the widened `8192` token budget.

### March 14, 2026 — `066d4e6` — calibration: validate GB10 defaults and add token-budget projections — score `3` — complexity `18`

**AI-identified within brief, human-shaped (3)**

- Validate the tightened calibration loop on the real FA4-backed GB10 path and close the remaining gap between the emitted operating point and the actual `300s` winner.
  - Meaning: the next slice is no longer about designing the projection and batch-audit machinery. It is about proving that the synced default path on GB10 now emits the right default, the right batch shape, and the right operator-facing UX.
  - Motivation: the just-landed control-loop changes are meaningful only if the next end-to-end GB10 run shows that winner-batch audit, friendly narration, and default no-`900s` behavior all work together on the real Spark path.
  - Purpose: turn the new calibration policy from a locally grounded design into a fully validated default workflow for the next cloned GB10 bring-up.
  - Replayed the finished FA4-backed `v18` GB10 bundle against the patched local code and fixed the two remaining calibration-control bugs: winner batch audit now prefers exact same-batch truth when available and ignores degenerate one-step audit curves, and the longer-horizon summary now keeps the standalone `300s` confirmation winner instead of recomputing the target winner from the `900s` run.
  - Promoted observed-at-target rows to first-class projection results in the shared curve layer so actual `300s` and `900s` confirmations are not damped back toward older truth anchors.
  - Made local search quality-aware by running its short probes with eval enabled, so the emitted default can no longer drift on pure throughput after winner-batch audit has settled the batch shape.
  - The replayed GB10 result now recovers the known good operating point for `m5-balanced`: winner batch audit selects `db=32`, `tb=32768`, while the merged longer-horizon summary says `m5-balanced` wins at `300s` and `m5-xlarge` wins at `900s`.
- Refactor the README front door so DGX Spark / GB10 is treated as a validated fast-track path alongside M4 / M5 instead of reading like a secondary CUDA appendix.
  - Meaning: the top-level repo story now explicitly has two validated fast tracks, then a generic bring-up path for everything else.
  - Motivation: the GB10 / Spark path is already real enough that burying it behind “generic CUDA” undersells the actual validated state of the project.
  - Purpose: make the README tell the same story the code and docs now support: MLX on M4/M5 and CUDA on DGX Spark / GB10 are both first-class on-ramps, while other hardware still goes through the more general calibration-first path.
  - Clean up the README `Project Structure` section into a grouped directory map so it reinforces the same top-level story instead of dumping a redundant flat file inventory.
  - Extend the DGX Spark setup guide with the grounded `900s` GB10 head-to-head reference so the doc now shows both the strict `300s` winner and the deeper longer-horizon crossover where `m5-xlarge` overtakes `m5-balanced`.
- Add a token-target calibration mode so the same projection and batch-audit loop can choose defaults against a fixed training-signal budget, with `100M` tokens as the default target.
  - Meaning: the shared curve machinery no longer assumes that calibration objectives are wall-clock horizons. The same truth-backed projection flow can now compare families at a fixed token budget, which is the first step toward the token-centric research path suggested by the Spark depth-scaling discussion.
  - Motivation: the GB10 `300s` and `900s` curves already showed that crossover behavior is easier to reason about in token space than in wall-clock space, especially when deeper or larger models process different numbers of tokens by the same time horizon.
  - Purpose: let calibration answer both “what wins at `300s`?” and “what wins by `100M` tokens?” without forking the projector, and keep the generic curve-report path aligned with the calibration runner.
  - Generalize `autoresearch_platform/curve_projection.py` so projections, truth matching, calibration residuals, and multi-horizon comparisons can target either seconds or tokens while preserving observed-at-target and truth-match behavior.
  - Wire `tools/calibrate_platform.py` and `tools/curve_report.py` through the same token-aware objective contract, including `--target-budget-mode tokens --target-tokens 100000000`, token-aware truth-curve loading, token-aware winner batch audit, and token-aware report labels while keeping the deeper seconds-based scaling confirmation as a separate opt-in path.
  - Add real `--token-budget` stopping support to `autoresearch_cuda/train.py`, then use it on GB10 to run the `m5-xlarge depth=12 db=16 tb=32768` permutation to a measured `50.0M` tokens under FA4. That run finished at `val_bpb=1.104588`, which beats the fixed-`50M` `m5-balanced` interpolation (`1.237246`) and slightly improves on the earlier `m5-xlarge` depth-8 projections (`1.122165` at `db=16/tb=32768`, `1.129174` at `db=32/tb=65536`).

**Grounding**

- Validation:
  - `python3 -m py_compile autoresearch_platform/curve_projection.py tools/calibrate_platform.py`
  - `python3 -m py_compile autoresearch_platform/curve_projection.py tools/calibrate_platform.py tools/curve_report.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/curve_report.py /tmp/gb10_token_truth/gb10_m5small_curve300.json /tmp/gb10_token_truth/scaling_m5-balanced_seq1024_db32_tb32768_SSSSL.curve.json /tmp/gb10_token_truth/scaling_m5-xlarge_seq2048_db32_tb65536_L.curve.json --truth-curves-dir /tmp/gb10_token_truth --target-budget-mode tokens --target-tokens 100000000`
  - `python3 tools/calibrate_platform.py --help`
  - replayed `/home/ent/cuda_fast_projection_fa4_v18` on the GB10 host with the patched code synced to `/home/ent/autoresearch-everywhere-sync`
- Measurements:
  - winner batch audit replay now picks `m5-balanced db=32 tb=32768` with exact-batch truth backing (`300s val_bpb=1.162382`)
  - merged longer-horizon replay now says `m5-balanced` at `300s` (`1.161863`) and `m5-xlarge` at `900s` (`1.094201`)
  - the new `100M` token projection replay over the GB10 truth corpus currently ranks:
    - `m5-xlarge` -> `1.146323`
    - `m5-balanced` -> `1.252032`
    - `m5-small` -> `1.317671`

### March 14, 2026 — `9a0d5e6` — calibration: turn truth-backed projection into a usable GB10 bring-up loop — score `3` — complexity `19`

**AI-identified within brief, human-shaped (3)**

- Turn the shared truth-backed horizon model into the real control loop for calibration instead of leaving it as a report-only appendix.
  - Meaning: family selection, finalist reruns, and longer-horizon reporting now all flow through the same token-accounted projection table, so the logic that explains the winner is also the logic that chooses the winner.
  - Motivation: once the GB10 truth corpus existed, the remaining gap was not better reporting. It was that `calibrate.py` could still make decisions using older heuristics and only explain them afterward with better projection tooling.
  - Purpose: make the `300s val_bpb` objective the center of the calibration loop and keep longer-horizon guidance explicit but secondary.
  - `tools/curve_report.py`, `autoresearch_platform/curve_projection.py`, and `tools/calibrate_platform.py` now share one multi-horizon table with next-horizon recommendations plus LR-style fit diagnostics such as damping, correction, fit quality, sigma/SNR, and horizon drift.
  - `calibrate.py` now uses that table to decide whether the first projection pass is already decisive, whether finalists need a longer rerun, and whether a separate longer-horizon scaling candidate should be surfaced.
  - The deeper `900s` scaling-confirmation path is now explicit and opt-in via `--enable-scaling-confirmation`, so the default path stays focused on the real `300s` objective while still reporting who looks strongest later.
- Correct batch selection so the emitted operating point is chosen for the same long-horizon objective as family selection, not for a short throughput plateau on the anchor preset.
  - Meaning: batch sizing is still anchored as a device-level first pass, but calibration now stops rewarding larger `total_batch_size` just because it sits near the throughput plateau and instead rechecks the selected family with a curve-aware audit before emitting an operating point.
  - Motivation: the GB10 FA4 evidence exposed a real policy bug. `m5-balanced` at `32 / 32768` beat `8 / 49152` on both `300s val_bpb` and total tokens, but the old short anchor probe still drifted toward the larger-`tb` shape.
  - Purpose: keep the cheap hardware-level batch profile, then correct it when the chosen family shows that optimizer-step cadence and accumulation cost matter more than a tiny short-run throughput difference.
  - The anchor selector now uses a `>= 90%` plateau and a `< 65%` cliff, stops preferring larger `tb` inside the plateau, and breaks ties toward lower accumulation/control cost before considering raw batch size.
  - After family selection, calibration now runs a dedicated `10s` winner batch audit on the chosen preset, revisiting both the current batch shape and the preset's broader candidate grid so better shapes like `32 / 32768` can re-enter even when the anchor drifted elsewhere.
  - That winner batch audit now ranks candidates by projected `300s val_bpb` first, then projected optimizer-step count, lower `grad_accum`, lower accumulation/control overhead, and finally throughput.
- Turn the GB10 / DGX Spark path into a reproducible FA4-backed bring-up and make calibration read like a front-door user workflow rather than an internal experiment harness.
  - Meaning: the docs and runtime metadata now describe one coherent GB10 story: clone the public repo, build the FA4 image, verify `flash_attn` plus `rustbpe`, run calibration, and then start the research loop from the calibrated result.
  - Motivation: the Spark path had become technically rich but narratively fragmented, with too much emphasis on side workflows and too little on the main user journey from first clone to a trustworthy calibrated CUDA default.
  - Purpose: make the first CUDA bring-up on GB10 reproducible, explain the environment assumptions clearly, and make the long-running `calibrate.py` command feel like an intentional operator-facing tool.
  - Blackwell runtime metadata now points GB10-class systems at the FA4 path (`Dao-AILab/flash-attention#2268`, `flash-attn4`) instead of the stale FlashAttention 3 labels.
  - `docs/dgx-spark-setup.md` now reads as a cohesive tutorial: clone `Entrpi/autoresearch-everywhere`, mutate `vllm-node-tf5:latest` into an FA4-capable `vllm-node-tf5-fa4:sm120` image with `rustbpe`, reserve `100g` SHM, run trainer smoke, run calibration, then launch the research loop; kernel-lab is now an appendix instead of the main bring-up path.
  - The Spark guide now includes a compact GB10 preset table grounded in the real `300s` runs and clearly distinguishes the emitted calibration operating point from the separate truth-curve comparison table so users do not confuse a short local-search probe with the full `300s` evidence.
  - `calibrate.py` now narrates its own progress by default, opening with what calibration is for, the expected phases and rough runtime, and then giving human-readable phase preambles and completion summaries unless `--plain-progress` is requested.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `autoresearch_cuda/runtime.py`
  - `autoresearch_platform/curve_projection.py`
  - `docs/cuda-core-loop-parity.md`
  - `docs/dgx-spark-setup.md`
  - `docs/kernel-lab.md`
  - `tools/curve_report.py`
  - `tools/calibrate_platform.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/curve_projection.py tools/curve_report.py tools/calibrate_platform.py`
  - `python3 tools/curve_report.py /tmp/gb10_curve_runs/gb10_m5small_curve60.json /tmp/gb10_curve_runs/gb10_m5balanced_curve60.json --truth-curves-dir /tmp/gb10_curve_runs --horizons 60,120,300,900 --json`
  - synthetic selector check covering `32 / 32768` vs `8 / 49152` through `select_best_batch_profile_row(...)` and `select_best_batch_audit_row(...)`
  - `python3 -m py_compile tools/calibrate_platform.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/render_autonomy_badge.py`
- Measurements:
  - the shared horizon table on the real GB10 `60s` `m5-small` vs `m5-balanced` pair now reports:
    - `m5-balanced` as the projected winner at `60s`, `120s`, and `300s`
    - winner probability `0.9731`
    - `300s` source `truth-match`
    - `900s` source `calibrated-extrapolation` with `confidence=low`
  - the generic next-horizon recommendation on that same pair resolves to `suggested_seconds = none` with reason `enough-signal` for the `300s` target.
  - the completed GB10 `300s` truth corpus that drove the new control policy is:
    - `m5-balanced db=32 tb=32768` -> `1.162382`
    - `m5-xlarge db=16 tb=32768` -> `1.169337`
    - `m5-small db=32 tb=32768` -> `1.257603`
    - `m5-large db=32 tb=32768` -> `1.285034`
    - `m5-tiny db=32 tb=32768` -> `1.465999`
  - the same truth corpus exposed the stale batch-policy failure that motivated the winner audit:
    - `m5-balanced db=32 tb=32768` -> `val_bpb=1.162382`, `total_tokens=74.9M`
    - `m5-balanced db=8 tb=49152` -> `val_bpb=1.204865`, `total_tokens=74.0M`
  - the synthetic selector checks now choose `32 / 32768` over `8 / 49152` both for the anchor throughput plateau and for the new winner batch-audit path.

### March 14, 2026 — `c09e78c` — calibration: add token-accounted multi-horizon projection reporting — score `3` — complexity `11`

**AI-identified within brief, human-shaped (3)**

- Add a multi-horizon projection gate that compares shorter and longer finalist curves against the same `300s` truth target before declaring that there is enough signal to stop.
  - Meaning: the shared curve layer now reasons across two observed horizons instead of trusting a single partial curve snapshot. The calibration flow compares the shorter finalist projections with the longer finalist projections, measures whether the winner stays stable as more real training time is observed, and only carries the `enough_signal` decision forward when the longer-horizon result is both strong enough and stable enough.
  - Motivation: the first GB10 truth-backed projection pass showed that a simple short-horizon rerank could still pick the wrong family. We needed a more disciplined way to ask whether an early winner was genuinely converging toward the `300s` winner or was just benefiting from startup behavior.
  - Purpose: make the horizon model trustworthy enough to guide early stopping decisions in `calibrate.py` without pretending that one short curve can stand in for the real `300s` objective.
  - Adding `MultiHorizonProjectionDiagnostics` and `MultiHorizonProjectionDecision` plus the shared `compare_multi_horizon_curves(...)` helper in `autoresearch_platform/curve_projection.py`.
  - Updating `tools/calibrate_platform.py` so the finalist stage compares the `30s` finalist curves against the earlier projection curves, records per-preset stability diagnostics, and gates `enough_signal` on that multi-horizon check instead of reading the long-horizon decision directly.
  - Generalizing `tools/curve_report.py` into a shared front end that can ingest arbitrary short-horizon and longer-horizon curve artifacts from files or directories, then emit one confidence-bearing projection table using the same shared projection logic as `calibrate.py`.
  - Preserving `truth-match` metadata through the winner-probability enrichment step in `autoresearch_platform/curve_projection.py`, so real multi-horizon reports keep their actual projection source and matched-truth count instead of silently degrading to `generic-projection`.
  - Switching the calibration and stability math from horizon-seconds to horizon-tokens where it matters, so truth-curve matching, calibration-point selection, and multi-horizon diagnostics all operate in the same token-accounted domain that the projector already used for its core fit.
  - Bringing over the remaining useful LR-style control diagnostics into the shared multi-horizon layer: minimum fit quality across horizons, effective damping, explicit horizon correction, projected gap, projected sigma, and projected SNR now sit alongside `alpha` and the older stability fields instead of being implicit inside the stop/go decision.
  - Extending the shared curve-report output to emit observed tokens, target tokens, calibration-token horizons, and token-based horizon diagnostics across arbitrary requested targets such as `60s`, `120s`, `300s`, and `900s`.
  - Wiring those same LR-style diagnostics through the calibration report path in `tools/calibrate_platform.py`, so the finalist diagnostics table and horizon tables expose the same fit-quality and correction signals as the standalone `curve_report.py` tool.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `autoresearch_platform/curve_projection.py`
  - `tools/curve_report.py`
  - `tools/calibrate_platform.py`
  - `docs/assets/autoresearch-everywhere.png`
- Validation:
  - `python3 -m py_compile autoresearch_platform/curve_projection.py tools/calibrate_platform.py`
  - synthetic positive multi-horizon projection check using `/tmp/test_multi_horizon.py`
  - synthetic stability-path probe using `/tmp/test_multi_horizon_unstable.py`
  - `python3 -m py_compile autoresearch_platform/curve_projection.py tools/curve_report.py tools/cuda_curve_report.py`
  - `python3 tools/curve_report.py /tmp/gb10_curve_runs/gb10_m5small_curve60.json /tmp/gb10_curve_runs/gb10_m5balanced_curve60.json --truth-curves-dir /tmp/gb10_curve_runs --horizons 60,120,300,900`
  - `python3 tools/curve_report.py /tmp/gb10_curve_runs/gb10_m5small_curve60.json /tmp/gb10_curve_runs/gb10_m5balanced_curve60.json --truth-curves-dir /tmp/gb10_curve_runs --json --horizons 60,120,300,900`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/render_autonomy_badge.py`
- Measurements:
  - the new multi-horizon path is grounded against the already completed GB10 `300s` truth corpus:
    - `m5-balanced` at `1.162382`
    - `m5-xlarge db=16` at `1.169337`
    - `m5-small` at `1.257603`
    - `m5-large` at `1.285034`
    - `m5-tiny` at `1.465999`
  - the synthetic positive check confirmed that a winner can remain `enough_signal=true` when the shorter and longer horizons agree closely on the projected target outcome
  - the shared token-accounted horizon table on the real GB10 `m5-small` vs `m5-balanced` `60s` curves now reports:
    - `60s`, `120s`, and `300s` as `truth-match`
    - `900s` as `calibrated-projection`
    - winner `m5-balanced` with projected winner-token counts of `10.9M`, `27.1M`, `75.5M`, and `236.9M` respectively

### March 14, 2026 — `9614d95` — calibration: add truth-backed CUDA horizon projection to bring-up — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Add a batch-profile phase ahead of every preset-family comparison, then reuse the tuned `device_batch_size` / `total_batch_size` everywhere instead of introducing a separate ranking batch regime.
  - Meaning: `calibrate.py` now runs `batch-profile` immediately after hardware fingerprinting, chooses one machine-shaped batch profile on an anchor preset, and reuses that same tuned batch for coarse envelope, ranking, projection, and finalist stages. The CUDA engine also now respects explicit batch-shape overrides during short probes instead of silently re-normalizing them away.
  - Motivation: on the M5 side we repeatedly found that machine batch and total-batch behavior were closer to hardware constants than preset-family constants, and the first GB10 FA4 bring-up showed the same risk on CUDA. Ranking presets at their shipped batch defaults was biasing family comparison before the real `300s` objective ever had a chance to speak.
  - Purpose: make bring-up compare model families under one tuned hardware batch regime, so the only thing left to optimize is which family gives the best `val_bpb` at the target horizon.
- Add a shared truth-curve projection path and the first real CUDA `300s` truth corpus so family selection can project the 5-minute winner from partial runs instead of guessing from short endpoints.
  - Meaning: the CUDA trainer now accepts `--curve-eval-seconds` and `--curve-output`, records periodic validation checkpoints without charging eval wall time against the training budget, and writes a machine-readable curve artifact. Those artifacts now feed the shared layer in `autoresearch_platform/curve_projection.py` plus generic reporting CLIs in `tools/curve_report.py` and `tools/cuda_curve_report.py`. The projection stage can match partial CUDA curves against completed `300s` truth curves on the same preset family and hardware bucket.
  - Motivation: the old `5s -> 10s -> 30s` reranks were still picking the wrong family on GB10. Real `300s` A/Bs under the tuned hardware batch showed that `m5-balanced` beats `m5-small` on the actual objective even when shorter horizons prefer smaller models.
  - Purpose: replace heuristic horizon correction with a truth-backed projection step that can eventually generalize across CUDA, MLX, ROCm, and future backends while staying faithful to the only objective that matters here: lowest `val_bpb` at `300s`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_platform/cuda_engine.py`
  - `autoresearch_platform/engines.py`
  - `autoresearch_platform/mlx_engine.py`
  - `autoresearch_cuda/train.py`
  - `autoresearch_platform/curve_projection.py`
  - `docs/cuda-core-loop-parity.md`
  - `tools/calibrate_platform.py`
  - `tools/curve_report.py`
  - `tools/cuda_curve_report.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/engines.py autoresearch_platform/mlx_engine.py autoresearch_platform/cuda_engine.py autoresearch_platform/curve_projection.py autoresearch_cuda/train.py tools/calibrate_platform.py tools/curve_report.py tools/cuda_curve_report.py`
  - real GB10 FA4-backed bring-up reruns completed with persistent outputs under:
    - `/home/ent/cuda_fast_projection_v7`
    - `/home/ent/curve_runs`
  - synthetic and real backend-agnostic curve-artifact summaries:
    - `python3 tools/cuda_curve_report.py /tmp/cuda_curve_synth.json --target-seconds 45`
    - `python3 tools/curve_report.py /home/ent/curve_runs/gb10_m5tiny_curve300.json /home/ent/curve_runs/gb10_m5small_curve300.json /home/ent/curve_runs/gb10_m5balanced_curve300.json --target-seconds 300`
- Measurements:
  - the tuned GB10 hardware batch profile used in the latest bring-up rerun is:
    - `device_batch_size=8`
    - `total_batch_size=40960`
    - `grad_accum_steps=10` on the anchor `m5-small`
  - full sequential GB10 `300s` truth set under tuned CUDA batch regimes:
    - `m5-balanced db=32 tb=32768` -> `1.162382`
    - `m5-xlarge db=16 tb=32768` -> `1.169337`
    - `m5-xlarge db=32 tb=65536` -> `1.170638`
    - `m5-small db=32 tb=32768` -> `1.257603`
    - `m5-large db=32 tb=32768` -> `1.285034`
    - `m5-tiny db=32 tb=32768` -> `1.465999`
  - corrected truth-matched projection from the saved `v7` partial curves:
    - `m5-balanced` -> projected `1.162382`
    - `m5-xlarge` -> projected `1.169395`
    - `m5-small` -> projected `1.257603`
    - `m5-large` -> projected `1.285034`
    - `m5-tiny` -> projected `1.465999`
  - this confirms the intended selector behavior: under the actual `300s val_bpb` objective, `m5-balanced` is the GB10 winner, while `m5-xlarge db=16` is the closest scaling candidate.

### March 13, 2026 — `0eb3fc5` — cuda/checkpoints: finish FA4-backed GB10 bring-up and normalize compiled checkpoint keys — score `2`

**AI-identified within brief, human-approved (2)**

- Fix CUDA checkpoint save/load so compiled-model checkpoints can be reused cleanly for eval calibration and resumed bring-up runs, then rerun the FA4-backed GB10 fast calibration to completion.
  - Meaning: CUDA checkpoints now strip the `_orig_mod.` prefix from compiled model state dict keys on save and normalize it again on load, so the exact checkpoint produced by a compiled trainer run can be resumed into an eager eval-only model. That removes the failure that previously stopped the FA4-backed GB10 fast bring-up after checkpoint minting and lets the full report, eval-rung bundle, and promotion artifacts complete.
  - Motivation: the first FA4-backed GB10 fast run had already proved that Blackwell should scale above `m5-tiny`, but the run died in the eval-calibration phase because the checkpoint was saved from a compiled model and then loaded into an eager model. That meant the actual platform-default result was present in partial artifacts but not cleanly promotable or documented.
  - Purpose: finish the real GB10/FA4 bring-up loop, ground the CUDA default on that machine, and remove a checkpoint-format mismatch that would otherwise keep biting compiled CUDA paths during calibration and resume workflows.
  - Updating the GB10 docs to record the completed FA4-backed result: `m5-small` is the recommended CUDA starting point on that system, with `seq_len=512`, `window_pattern=L`, `device_batch_size=32`, and `total_batch_size=32768`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/cuda-core-loop-parity.md`
  - `autoresearch_cuda/checkpoints.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/checkpoints.py`
  - real GB10 eval-only rerun against the previously failing candidate checkpoint:
    - `python train.py --engine cuda --preset m5-small --resume-from ... --eval-only --eval-seq-len 2048 --eval-tokens 262144 --eval-batch-size 32 --no-compile`
  - resumed FA4-backed fast bring-up on GB10 without `--force`, reusing the already completed earlier phases:
    - `python calibrate.py --engine cuda --mode fast --output-dir .../cuda_fast_ladder_fa4`
- Measurements:
  - the direct resumed eval-only check now succeeds against the compiled checkpoint:
    - `cheap` rung `val_bpb=1.282243`
    - `eval_seconds=0.8`
    - `resolved_attention_backend=installed:flash_attn.flash_attn_interface`
  - the completed FA4-backed GB10 fast bring-up selected:
    - candidate default preset `m5-small`
    - tuned point `seq_len=512`, `window_pattern=L`, `device_batch_size=32`, `total_batch_size=32768`, `grad_accum_steps=2`
    - ranking `val_bpb=1.291656`
    - ranking steady throughput `541924.6 tok/s`
  - completed reduced eval rung table from the FA4-backed fast bring-up:
    - `cheap`: `val_bpb=1.282243`, `eval_seconds=0.8`
    - `reference`: `val_bpb=1.253053`, `eval_seconds=4.5`
  - full bundle written under `/home/ent/autoresearch-everywhere-sync/results/analysis/cuda_fast_ladder_fa4/`, including `report.json`, `report.md`, and `promotion/platform_default.json`

### March 13, 2026 — `d152d06` — calibration: align CUDA bring-up with the shared ladder and validate checkpoint-backed eval calibration on GB10 — score `2`

**AI-identified within brief, human-approved (2)**

- Add and validate the next CUDA parity slice: shared-ladder bring-up plus checkpoint-backed eval-only rung execution through the shared engine on a real GB10 box.
  - Meaning: the CUDA path now uses the same named model-scale ladder as the MLX side (`m5-tiny` through `m5-xlarge`) for platform bring-up, with bounded short-probe batch shapes that keep Blackwell calibration runs practical. The CUDA trainer can also resume from an exact checkpoint in an `--eval-only` mode, override eval sequence length / token budget / batch size, and emit structured eval-only summaries. The shared CUDA engine can call that path repeatedly to build `cheap` / `reference` / `full` rung reports against a saved checkpoint instead of only talking about future parity in the abstract.
  - Motivation: exact checkpoint/resume was already landed, but the parity roadmap still honestly said CUDA had no real eval-calibration story, and the bring-up path was still anchored to CUDA-specific shapes rather than the shared preset ladder. The next thing worth proving was not more theory; it was whether the shared engine could use the common scale ladder, drive rung evaluation on the real GB10 system, and return useful candidate-default artifacts.
  - Purpose: move CUDA from “trainer can resume” to “trainer can participate in the same checkpoint-backed calibration workflow shape as MLX,” while also making `calibrate.py --engine cuda` behave like a real shared-platform bring-up instead of a special-case CUDA probe.
  - Defining a CUDA-side `m5-tiny` → `m5-xlarge` ladder in `autoresearch_cuda/config.py`.
  - Extending `autoresearch_cuda/train.py` with configurable `--eval-only`, `--eval-seq-len`, `--eval-tokens`, and `--eval-batch-size`.
  - Extending `autoresearch_platform/cuda_engine.py` with `run_eval_calibration(...)`.
  - Teaching the shared CUDA engine to use bounded short-probe batch shapes and complete fast-mode bring-up through checkpoint mint, local search, and eval calibration.
  - Fixing markdown artifact creation so eval-calibration reports can be written into fresh result directories.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/cuda-core-loop-parity.md`
  - `autoresearch_cuda/config.py`
  - `autoresearch_cuda/prepare.py`
  - `autoresearch_cuda/train.py`
  - `autoresearch_platform/cuda_engine.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/config.py autoresearch_cuda/prepare.py autoresearch_cuda/train.py autoresearch_platform/cuda_engine.py`
  - real GB10 fast bring-up:
    - `python calibrate.py --engine cuda --mode fast ...`
  - real GB10 eval-only rung:
    - `python train.py --engine cuda --preset upstream --resume-from ... --eval-only --eval-seq-len 2048 --eval-tokens 262144 --eval-batch-size 32 --no-checkpoint`
  - real GB10 shared-engine rung calibration:
    - `CUDAEngine.run_eval_calibration(..., rungs=["cheap", "reference", "full"], ...)`
- Measurements:
  - GB10 fast-mode bring-up on the minimal CUDA container completed end to end through checkpoint mint and local search and initially selected:
    - candidate default preset `m5-tiny`
    - tuned point `seq_len=256`, `device_batch_size=16`, `total_batch_size=8192`, `grad_accum_steps=2`
  - GB10 full rung table from the shared engine:
    - `cheap`: `val_bpb=2.212186`, `eval_seconds=1.0`, `abs_error_vs_full=0.038056`
    - `reference`: `val_bpb=2.161972`, `eval_seconds=5.5`, `abs_error_vs_full=0.012158`
    - `full`: `val_bpb=2.174130`, `eval_seconds=72.2`
  - markdown artifact written successfully at `/home/ent/autoresearch-everywhere/results/analysis/gb10_eval_calibration.md`
  - the validated rung run resolved `torch-sdpa` in the minimal GB10 container, so these numbers prove the calibration path itself rather than the separately validated FA4-enabled path

### March 13, 2026 — `54c574a` — cuda: add exact checkpoint and resume support — score `2`

**AI-identified within brief, human-approved (2)**

- Add exact sync checkpoint minting and exact resume to the core CUDA trainer loop, then validate it on the real GB10 Blackwell system.
  - Meaning: the CUDA trainer can now write a final exact checkpoint bundle containing model weights, optimizer state, run config, and training progress, and then resume from that bundle while restoring the same preset shape and replaying the train-loader position deterministically. Resumed runs interpret `--time-budget` as a new cumulative target, not an extra delta, and the trainer no longer counts the first resumed compile-heavy step against the resumed training budget.
  - Motivation: the CUDA parity roadmap was no longer blocked on “can it run?” GB10 had already proven that. The next missing trainer-loop feature was exact resumability, because without it CUDA still could not match the baseline durability and checkpoint-backed workflow expected from the MLX path.
  - Purpose: close the basic survivability gap in the CUDA core loop so later parity work can build on a trainer that can actually mint and reuse checkpoints instead of restarting every long run from scratch.
  - Adding `autoresearch_cuda/checkpoints.py` for exact bundle save/load and metadata validation.
  - Teaching `autoresearch_cuda/train.py` to support `--checkpoint-path`, `--resume-from`, cumulative resume budgets, and deterministic loader replay.
  - Extending `autoresearch_platform/cuda_engine.py` so checkpoint minting works through the shared train-probe path.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/cuda-core-loop-parity.md`
  - `autoresearch_cuda/checkpoints.py`
  - `autoresearch_cuda/train.py`
  - `autoresearch_platform/cuda_engine.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/checkpoints.py autoresearch_cuda/train.py autoresearch_platform/cuda_engine.py tools/calibrate_platform.py`
  - real GB10 checkpoint mint:
    - `python train.py --engine cuda --preset upstream --time-budget 2 --seq-len 1024 --window-pattern SSSL --device-batch-size 32 --total-batch-size 65536 --benchmark-skip-eval --checkpoint-path ...`
  - real GB10 resume:
    - `python train.py --engine cuda --preset upstream --resume-from ... --time-budget 4 --benchmark-skip-eval`
- Measurements:
  - GB10 exact checkpoint mint:
    - `training_seconds=2.0`
    - `total_seconds=33.0`
    - `peak_vram_mb=6148.7`
    - `steady_state_tok_per_sec=130348.1`
    - `num_steps=14`
    - `resolved_attention_backend=torch-sdpa`
  - GB10 exact resume after timing fix:
    - `training_seconds=4.0`
    - `total_seconds=26.0`
    - `peak_vram_mb=6148.7`
    - `steady_state_tok_per_sec=65388.2`
    - `num_steps=18`
    - `resolved_attention_backend=torch-sdpa`
    - resumed from the saved checkpoint with deterministic loader replay (`loader_batches=28`)

### March 13, 2026 — `b8871fe` — calibration: start CUDA engine parity with local search and runtime-safe bring-up — score `2`

**AI-identified within brief, human-approved (2)**

- Start the first real CUDA core-loop parity slice by making the shared CUDA engine participate in local search and by removing MLX-specific import assumptions that were breaking CUDA-only bring-up environments.
  - Meaning: the CUDA engine now advertises and implements local search through the shared engine boundary, with real sequence-length, window-pattern, and batch-shape candidates instead of a single fixed upstream point. `calibrate_platform.py` also no longer hard-imports MLX eval-policy modules at startup, so `calibrate.py --engine cuda` can run in a CUDA-only environment without `mlx` installed. The CUDA trainer now treats the repo-local `kernels` package as optional instead of a mandatory import, which keeps GB10-class environments that rely on installed FlashAttention packages from failing before training even begins.
  - Motivation: the CUDA parity roadmap was already committed, and the next concrete gap was clear: `calibrate.py --engine cuda` still could not behave like a real bring-up flow because the shared engine exposed no local search and the orchestration stack still assumed MLX was importable. The GB10 validation environment also exposed that the trainer was too brittle about where FlashAttention came from.
  - Purpose: turn the parity roadmap into the first real code step toward a CUDA engine that can participate in the same bring-up loop shape as MLX, while making the CUDA path robust in the GB10 container/runtime environment we are actively validating against.
  - Extending `autoresearch_platform/cuda_engine.py` so full-mode CUDA bring-up can search beyond one fixed upstream shape.
  - Making `tools/calibrate_platform.py` degrade cleanly on non-MLX hosts instead of failing on top-level imports.
  - Making `autoresearch_cuda/train.py` accept environments where installed FlashAttention is present but the repo-local `kernels` package is not.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_cuda/train.py`
  - `autoresearch_platform/cuda_engine.py`
  - `tools/calibrate_platform.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/train.py autoresearch_platform/cuda_engine.py tools/calibrate_platform.py`
  - local probes confirming CUDA engine local-search candidates and non-MLX import safety
  - remote GB10 container bring-up progressed through hardware fingerprinting into the first coarse CUDA trainer probe instead of failing immediately on MLX imports or missing repo-local kernels
- Measurements:
  - No new stable trainer benchmark was recorded in this slice; the main grounded result is that the parity path now reaches real CUDA probing on GB10 instead of stopping on infrastructure mismatches.

### March 13, 2026 — `69960ca` — docs: add CUDA core-loop parity roadmap — score `2`

**AI-identified within brief, human-approved (2)**

- Add a dedicated CUDA parity assessment that explains what already works, what is still missing from the shared trainer loop, and how GB10, A100, H100, and B200 should be used to close the gap.
  - Meaning: the repo now has a standalone roadmap for CUDA trainer parity in `docs/cuda-core-loop-parity.md`. It defines what “parity” means here, separates already-strong CUDA areas from the remaining MLX-only features, documents what GB10 already proved, and lays out the phased path toward checkpointing, eval calibration, runtime policy, platform bring-up, kernel-lab trainer integration, and finally whole attention backend experiments such as FlashAttention and SageAttention.
  - Motivation: the repo had enough real CUDA surface area and GB10 evidence that the next problem was no longer “can CUDA run?” It was “what exactly still separates CUDA from the MLX-first calibrated trainer loop, and what is the right order to close those gaps across the NVIDIA hardware matrix?”
  - Purpose: make the CUDA path legible as a deliberate parity effort rather than a pile of independent runtime, lab, and hardware-specific improvements.
  - Linking that roadmap from both the main README CUDA shortcut and the kernel-lab roadmap so it is part of the front-door story rather than an orphaned internal note.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/cuda-core-loop-parity.md`
  - `docs/kernel-lab.md`
- Validation:
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - No new runtime measurements; this is a synthesis and planning document grounded in the existing GB10 validation and the current shared-engine code.

### March 13, 2026 — `71788a0` — cuda: add GB10-compatible attention fallback and shared summary parsing — score `2`

**AI-identified within brief, human-approved (2)**

- Make the CUDA trainer resolve real FlashAttention implementations more flexibly and fall back cleanly when they are unavailable, while moving trainer-summary parsing into one shared module.
  - Meaning: the CUDA trainer no longer assumes a single packaged FlashAttention path. It now tries the installed `flash_attn.flash_attn_interface` path first, then the older `hopper.flash_attn_interface`, then the repo-local `kernels` package, and finally falls back to PyTorch SDPA with a local-window causal mask. At the same time, the `parse_summary` helper is no longer duplicated inside the eval-calibration tooling; it now lives in `autoresearch_platform/summary.py` and is reused by both MLX and CUDA platform code.
  - Motivation: the real GB10 validation run used FlashAttention 4 from the SM120-support PR, which imports through the installed `flash_attn.flash_attn_interface` path rather than the older repo-assumed layout. Without this fallback stack, the trainer stayed artificially brittle even after the GB10 environment was otherwise working. The duplicated summary parser was also an unnecessary divergence across platform tooling.
  - Purpose: keep the CUDA trainer runnable across real packaged FlashAttention variants on Blackwell/GB10 and simplify the shared engine/calibration code before a deeper CUDA core-loop parity push.
  - Surfacing both the preferred attention backend and the resolved runtime backend in trainer output so GB10 runs make the actual path explicit.
  - Repointing MLX/CUDA engine utilities and eval-calibration helpers to one shared summary parser.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `autoresearch_cuda/train.py`
  - `autoresearch_mlx/lab_workspace.py`
  - `autoresearch_platform/cuda_engine.py`
  - `autoresearch_platform/mlx_engine.py`
  - `autoresearch_platform/summary.py`
  - `tools/calibrate_eval_policy.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/train.py autoresearch_platform/summary.py autoresearch_platform/cuda_engine.py autoresearch_platform/mlx_engine.py tools/calibrate_eval_policy.py autoresearch_mlx/lab_workspace.py`
- Measurements:
  - Real GB10 CUDA smoke after syncing the fallback logic:
    - `resolved_attention_backend=installed:flash_attn.flash_attn_interface`
    - `steady_state_tok_per_sec=245727`
    - `peak_vram_mb=45011.5`
    - `num_params_M=50.3`

### March 13, 2026 — `fde764c` — docs: add a kernel-lab roadmap toward whole attention backends — score `2`

**AI-identified within brief, human-approved (2)**

- Add a roadmap section to `docs/kernel-lab.md` that evaluates the current lab state and lays out the path from seam-level kernels to whole attention backends such as FlashAttention and SageAttention.
  - Meaning: the kernel-lab doc no longer stops at describing the current workflows. It now ends with an explicit assessment of where the lab is strong today, where it is still weak, and the staged path from starter seams to composed-path work, backend-level adapters, whole attention backend experiments, and eventual cross-backend parity.
  - Motivation: the lab has grown beyond a starter-kernel catalog, but the docs did not yet explain how the current MLX and CUDA systems fit into a larger plan. Without that roadmap, it is easy to either overread the current maturity or jump too quickly from seam-level wins to backend-sized ambitions like FlashAttention or SageAttention.
  - Purpose: make the next steps legible, so kernel-lab reads as a growing subsystem with a deliberate direction rather than an accumulation of isolated targets and commands.
  - Refreshing the top of `docs/kernel-lab.md` at the same time so it no longer understates CUDA by calling Triton workspaces purely future work.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `docs/kernel-lab.md`
- Validation:
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - No new runtime measurements; this was a documentation and planning update.

### March 13, 2026 — `0c5657b` — docs: fold CUDA into the main README flow — score `2`

**AI-identified within brief, human-approved (2)**

- Fold CUDA into the main README start-here story instead of leaving it in an appendix, and keep the workstation sweep section framed around its actual user story.
  - Meaning: the README now presents CUDA as a first-class engine behind the same `prepare.py`, `train.py`, `calibrate.py`, and `kernel-lab.py` front doors, with a dedicated NVIDIA shortcut under `Start Here`, instead of treating it as a side note after the main story. The sweep tooling section also stays framed as `Manual Longer Sweeps`, which is the real user-facing job it does.
  - Motivation: the appendix framing made CUDA feel secondary even though the repo already has a real shared-engine boundary, a CUDA runtime layer, GB10 validation, and a serious kernel-lab path. At the same time, the old “optional tooling” language hid the point of the longer-sweep scripts behind implementation-sounding wording.
  - Purpose: make the README read like the product we actually have: one cross-platform front door with MLX and CUDA as part of the same story, plus a clearly named manual sweep path for longer workstation runs.
  - Adding a top-level CUDA shortcut that explains what the NVIDIA path can already do today and where it differs from MLX.
  - Removing the old CUDA appendix once that information was folded into the main flow.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
- Validation:
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - No new runtime measurements; this was a documentation and framing cleanup.

### March 13, 2026 — `b74bff3` — docs: tighten the kernel-lab front door — score `2`

**AI-identified within brief, human-approved (2)**

- Rewrite the README kernel-lab section around the user-facing workflow and capabilities story instead of the full command catalog.
  - Meaning: the front door now explains what kernel-lab is for, what it can do on MLX and CUDA today, and how to start from it, while pushing the long command lists, GB10 setup notes, and deeper backend mechanics into the linked kernel-lab doc.
  - Motivation: the old README section had become a detailed subsystem dump. It was accurate, but too long and too procedural for a front-door reader trying to understand whether kernel-lab is relevant to them.
  - Purpose: make the top-level README better at selling the kernel-lab UX and current capability envelope without making users absorb the entire implementation story up front.
  - Reframing the section around:
    - safe kernel experimentation
    - shared cross-backend workflow
    - current MLX-vs-CUDA strengths
    - one or two concrete starting commands
  - Moving the detailed workflows and setup notes behind [docs/kernel-lab.md](/Users/ent/Codex/autoresearch/docs/kernel-lab.md).

### March 13, 2026 — `827b199` — docs: pin GB10 CUDA validation environment — score `2`

**AI-identified within brief, human-approved (2)**

- Tighten CUDA trace-family parsing and document the first real GB10 deep-profile results and setup requirements.
  - Meaning: the CUDA lab is no longer only synthetically trace-capable. It now has a real Blackwell GB10 validation path where `capture`, `trace-profile`, `auto-review`, and `deep-profile` all run successfully, and the lab’s family ranking is corrected against actual Nsight Systems output instead of the earlier false-positive `matmul_epilogue` mapping.
  - Motivation: once host GPU counters were enabled, the remaining blocker was no longer NVIDIA tooling. It was our own parser and matching logic. The first real GB10 run showed that we were misclassifying random/fill/sin/cos setup kernels and that Nsight Compute on this host emits a wide CSV format our parser did not understand.
  - Purpose: make the CUDA trace-first lab genuinely trustworthy on real hardware before pushing harder on trainer integration or automatic promotion.
  - Updating `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py` so:
    - copy/cast kernels rank under `data_movement` instead of being swallowed by `launch_fusion`
    - generic setup/elementwise kernels rank under `launch_fusion` instead of falsely surfacing as `matmul_epilogue`
    - `deep-profile` supports both long and wide Nsight Compute CSV formats
  - Updating `README.md` and `docs/kernel-lab.md` so the docs now record:
    - the real GB10 result (`launch_fusion` about `65.4%`, `data_movement` about `34.6%`, dominant issue `sync-bound`)
    - the current promotion state (`launch_fusion` ready for a real CUDA workspace, `data_movement` still blocked on manual review because deep diagnosis is weak/mixed)
    - the host-side setup needed for full CUDA profiling (`NVreg_RestrictProfilingToAdminUsers=0`, reboot, and `--cap-add=SYS_ADMIN` in the container path), the exact FlashAttention 4 install source used on GB10 via the SM120-support PR, and the concrete container snapshot used for the validation run (the local March 1, 2026 image of `vllm-node-tf5:latest`, image ID `sha256:c1ba011f841cacdfc234e5b754b1cb5e8120b8d4bd6b896c6703b28a44ba185a`, source repo `eugr/spark-vllm-docker`, best available source pin `8f11e7e5edd8c964f7a44fbd29f0c86a8df49a82`, NVIDIA PyTorch `26.01` build `256811084` / ref `9fa5c48351cf93ac6e6972ca113a7e3c54675a76`, PyTorch `2.10.0a0+a36e1d39eb.nv26.01.42222806`, CUDA `13.1`)

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_trace.py`
  - real GB10 `deep-profile` rerun for `data_movement` after enabling GPU counters and patching wide-CSV parsing:
    - `status=ok`
    - `diagnosis=mixed`
    - `confidence=0.45`
  - real GB10 `deep-profile` for `launch_fusion`
  - real GB10 `evidence` / `promotion-check` on both `launch_fusion` and `data_movement`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - Real GB10 trace-profile after the matcher fix:
    - `launch_fusion`: `65.44129853361615%`
    - `data_movement`: `34.55870146638384%`
    - dominant issue: `sync-bound`
  - Real GB10 deep-profile for `data_movement`:
    - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: `1.59`
    - `smsp__warps_active.avg.pct_of_peak_sustained_active`: `91.88`
    - diagnosis: `mixed`

### March 12, 2026 — `0f3d248` — lab: add Triton-backed CUDA attention prelude workspace — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Promote `attention_prelude` from a reference-first CUDA target into a Triton-optional starter workspace.
  - Meaning: `attention_prelude` no longer stops at the pure reference path. It now has a real optional Triton pointwise implementation for the value-embed gate-application stage inside the broader Q/K/V staging path, while keeping the existing fixed starter harness and reference projection path.
  - Motivation: after the pointwise, RoPE-side, loss-side, and MLP-side Triton slices, the next practical step was the broader attention staging seam. The honest first cut is the repeated gate-application subpath, not a fake full Triton rewrite of the whole attention-prelude stack.
  - Purpose: keep extending Triton support incrementally with honest starter workspaces that accelerate real repeated seams inside larger block-local paths without overselling full attention-side coverage.
  - Extending `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with an optional Triton gate-application kernel and marking `attention_prelude` as `triton-optional`.
  - Updating `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py`, `README.md`, and `docs/kernel-lab.md` so the CUDA starter catalog now reports `attention_prelude` as part of the Triton-backed slice.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target attention_prelude --workspace /tmp/cuda-triton-attention-prelude`
  - `python3 -m py_compile /tmp/cuda-triton-attention-prelude/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-attention-prelude --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-attention-prelude --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the new Triton-optional starter workspace still degrades cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.

### March 12, 2026 — `bbc9a01` — lab: add Triton-backed CUDA fused MLP workspace — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Promote `fused_mlp` from a reference-first CUDA target into a Triton-optional starter workspace.
  - Meaning: `fused_mlp` no longer stops at the pure reference path. It now has a real optional Triton pointwise implementation for the squared-ReLU activation stage, while keeping the existing fixed starter harness and reference GEMM path.
  - Motivation: after the pointwise, RoPE-side, and loss-side Triton slices, the next practical step was an MLP-side seam that is still repeated and trace-visible but does not pretend we already have a full Triton GEMM replacement story.
  - Purpose: keep extending Triton support incrementally with honest starter workspaces that accelerate real repeated subpaths without overselling full fused-block coverage.
  - Extending `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with an optional Triton squared-ReLU activation kernel and marking `fused_mlp` as `triton-optional`.
  - Updating `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py`, `README.md`, and `docs/kernel-lab.md` so the CUDA starter catalog now reports `fused_mlp` as part of the Triton-backed slice.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target fused_mlp --workspace /tmp/cuda-triton-fused-mlp`
  - `python3 -m py_compile /tmp/cuda-triton-fused-mlp/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-fused-mlp --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-fused-mlp --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the new Triton-optional starter workspace still degrades cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.

### March 12, 2026 — `998f752` — lab: add Triton-backed CUDA loss prelude workspace — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Promote `loss_prelude` from a reference-first CUDA target into a Triton-optional starter workspace.
  - Meaning: `loss_prelude` no longer stops at the pure reference path. It now has a real optional Triton row-wise implementation for the softcapped cross-entropy-prelude path, while keeping the existing fixed starter harness.
  - Motivation: after the narrower pointwise and RoPE-side Triton targets, the next practical step was a loss-side row kernel that is still trace-visible and benchmarkable without pretending the whole logits/loss stack is already fused.
  - Purpose: keep extending Triton support incrementally with honest starter workspaces that match the real scope of what has been optimized so far.
  - Extending `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with an optional Triton row-wise cross-entropy-prelude kernel and marking `loss_prelude` as `triton-optional`.
  - Updating `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py`, `README.md`, and `docs/kernel-lab.md` so the CUDA starter catalog now reports `loss_prelude` as part of the Triton-backed slice.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target loss_prelude --workspace /tmp/cuda-triton-loss-prelude`
  - `python3 -m py_compile /tmp/cuda-triton-loss-prelude/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-loss-prelude --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-loss-prelude --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the new Triton-optional starter workspace still degrades cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.

### March 12, 2026 — `602b2f8` — lab: add Triton-backed CUDA rope/QK starter workspace — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Promote `rope_qk_fused` from a reference-first CUDA seam into a Triton-optional starter workspace.
  - Meaning: `rope_qk_fused` no longer stops at the pure reference path. It now has a real optional Triton row-wise implementation for the fused RoPE + RMSNorm work, while keeping the existing trainer-side seam and fixed starter harness.
  - Motivation: after `value_embed_gate`, the next best Triton candidate was another narrow attention-side seam that is still pointwise/reduction-shaped and already has a direct trainer hook, without overclaiming that the lab can optimize the full attention core yet.
  - Purpose: keep extending Triton support incrementally with honest starter workspaces that match the real scope of what has been optimized so far.
  - Extending `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with an optional Triton row-wise RoPE + RMSNorm kernel and marking `rope_qk_fused` as `triton-optional`.
  - Updating `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py`, `README.md`, and `docs/kernel-lab.md` so the CUDA starter catalog now reports `rope_qk_fused` as part of the Triton-backed slice.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target rope_qk_fused --workspace /tmp/cuda-triton-rope-qk-fused`
  - `python3 -m py_compile /tmp/cuda-triton-rope-qk-fused/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-rope-qk-fused --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-rope-qk-fused --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the new Triton-optional starter workspace still degrades cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.

### March 12, 2026 — `6f93a5e` — lab: add Triton-backed CUDA value-embed gate workspace — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Promote `value_embed_gate` from a reference-first CUDA seam into a truthful Triton-optional starter workspace.
  - Meaning: `value_embed_gate` still uses the reference gate projection, but it now has a real optional Triton kernel for the hot pointwise gate-application path. The target is no longer just trainer-hookable; it now has a credible backend-specific workspace implementation too.
  - Motivation: after landing the narrow trainer seam, the next practical Triton step was to accelerate the repeated pointwise portion without pretending the lab already has a fully fused GEMM-plus-gating kernel.
  - Purpose: keep extending Triton support incrementally with honest starter workspaces that match the real scope of what has been optimized so far.
  - Extending `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with an optional Triton pointwise gate-application kernel and marking `value_embed_gate` as `triton-optional`.
  - Updating `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py`, `README.md`, and `docs/kernel-lab.md` so the CUDA starter catalog now reports `value_embed_gate` as part of the Triton-backed slice.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target value_embed_gate --workspace /tmp/cuda-triton-value-embed-gate`
  - `python3 -m py_compile /tmp/cuda-triton-value-embed-gate/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-value-embed-gate --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-value-embed-gate --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the new Triton-optional starter workspace still degrades cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.

### March 12, 2026 — `801c84d` — lab: add CUDA value-embed gate trainer seam — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Add the next narrow CUDA trainer seam by splitting value-embedding gating out of the broader attention-prelude path.
  - Meaning: the CUDA lab can now exercise a dedicated `value_embed_gate` seam inside attention staging, instead of only handling that work as an opaque part of the broader `attention_prelude` family. The new target is starter-ready, reference-first, and directly hookable in the real trainer path.
  - Motivation: once `attention_prelude` and `rope_qk_fused` were in place, the next useful trainer-side seam was the value-embed gate itself. It is repeated, narrow, and much easier to benchmark and promote than a larger attention-side fused region.
  - Purpose: keep broadening the CUDA direct-hook set with narrow, trainer-relevant seams that can eventually justify Triton work, instead of jumping straight to heavier attention-core kernels.
  - Extending `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with a starter-ready `value_embed_gate` target and fixed harness.
  - Wiring `/Users/ent/Codex/autoresearch/autoresearch_cuda/train.py` and `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_integration.py` so `value_embed_gate` can run through the real trainer path.
  - Updating the CUDA target catalog/docs so the direct-hook starter set now includes `value_embed_gate`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_integration.py`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_integration.py autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py autoresearch_cuda/train.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target value_embed_gate --workspace /tmp/cuda-value-embed-gate`
  - `python3 -m py_compile /tmp/cuda-value-embed-gate/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-value-embed-gate --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-value-embed-gate --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-value-embed-gate --preset upstream --time-budget 2 --benchmark-skip-eval --no-checkpoint`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the new starter workspace and trainer hook still degrade cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.

### March 12, 2026 — `f4f7a91` — lab: add Triton-backed CUDA logits softcap hook — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Add the next narrow CUDA trainer seam by wiring logits softcap into both the starter workspace catalog and the real trainer path.
  - Meaning: the CUDA lab can now exercise one more honest end-to-end seam after the output projection but before loss handling. `logits_softcap` is now a starter-ready CUDA workspace family, has a direct trainer-side hook, and ships with an optional Triton pointwise implementation rather than only a reference path.
  - Motivation: after the first Triton-backed CUDA starter families and their trainer hooks were in place, the next practical addition was a narrow logits-side pointwise seam that is easier to benchmark and promote than deeper attention-core work.
  - Purpose: broaden the real Triton-backed CUDA direct-hook set incrementally, so the CUDA lab keeps gaining trainer-relevant targets without jumping straight to much heavier kernels.
  - Extending `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with a starter-ready `logits_softcap` target and optional Triton implementation.
  - Wiring `/Users/ent/Codex/autoresearch/autoresearch_cuda/train.py` and `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_integration.py` so `logits_softcap` can run through the real trainer path.
  - Updating the CUDA target catalog/docs so the Triton-backed starter set and direct-hook set both include `logits_softcap`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_integration.py`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_integration.py autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py autoresearch_cuda/train.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target logits_softcap --workspace /tmp/cuda-triton-logits-softcap`
  - `python3 -m py_compile /tmp/cuda-triton-logits-softcap/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-logits-softcap --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-logits-softcap --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-triton-logits-softcap --preset upstream --time-budget 2 --benchmark-skip-eval --no-checkpoint`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the new starter workspace and trainer hook still degrade cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.

### March 12, 2026 — `22f29c8` — lab: mark Triton-backed CUDA starter metadata consistently — score `2`

**AI-identified within brief, human-approved (2)**

- Tighten the follow-up metadata and changelog grounding after the broader Triton starter-family expansion landed.
  - Meaning: generated CUDA starter workspaces now report the same Triton-backed implementation status in `metadata.json` that the templates and docs already claimed, so the starter catalog is internally consistent again.
  - Motivation: after the broader Triton-backed slice landed, `data_movement` and `matmul_epilogue` still emitted `workspace_impl=reference` in generated metadata even though the workspace templates had already been upgraded to optional Triton paths.
  - Purpose: keep the CUDA starter-catalog state machine honest before building more trainer-side CUDA lab depth on top of it.
  - Updating `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` so generated `data_movement` and `matmul_epilogue` workspaces are marked `triton-optional` just like the underlying templates.
  - Moving the already-landed broader Triton starter-family change into committed history and recording the actual validation that was run for it.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target data_movement --workspace /tmp/cuda-triton-data-movement`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-data-movement --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-matmul-epilogue --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the starter workspaces still degrade cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.

### March 12, 2026 — `c8f74d3` — lab: extend Triton-backed CUDA starter workspaces — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Extend the first Triton-backed CUDA slice from the narrow launch-fusion and RMSNorm targets into the next practical starter workspace families.
  - Meaning: the CUDA lab is starting to move from “a couple of proof-point Triton starters” toward a broader starter catalog where more trace-ranked families can open a real Triton workspace instead of only a reference workspace.
  - Motivation: once the first Triton-backed starters existed, the next useful step was to cover the adjacent families that are still narrow enough for a fixed harness but are closer to real trainer-side CUDA work, starting with `data_movement` and `matmul_epilogue`.
  - Purpose: broaden the real Triton-backed CUDA substrate without skipping straight to giant attention or GEMM rewrites, so the trace-first CUDA lab can keep turning ranked families into concrete backend workspaces incrementally.
  - Extending `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with optional Triton implementations for `data_movement` and `matmul_epilogue`.
  - Updating the CUDA lab docs and target notes so they keep distinguishing Triton-backed starter targets from reference-first starter targets honestly.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_workspace.py autoresearch_cuda/lab_trace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target data_movement --workspace /tmp/cuda-triton-data-movement`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target matmul_epilogue --workspace /tmp/cuda-triton-matmul-epilogue`
  - `python3 -m py_compile /tmp/cuda-triton-data-movement/kernel.py /tmp/cuda-triton-matmul-epilogue/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-data-movement --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-matmul-epilogue --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the new Triton-backed starter workspaces still degrade cleanly to `missing-runtime` because PyTorch/CUDA are unavailable.
  - The Triton-backed CUDA starter set is now:
    - `launch_fusion`
    - `norm`
    - `data_movement`
    - `matmul_epilogue`
  - The rest of the CUDA starter catalog remains reference-first for now.

### March 12, 2026 — `2ea3d54` — lab: add first Triton-backed CUDA starter workspaces — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Added the first real Triton-backed CUDA workspace implementations, but kept the slice intentionally narrow so the existing fixed harness and trace-first CUDA workflow stay honest.
  - Meaning: the CUDA lab is no longer only a reference starter-harness plus trace automation story. Two starter-ready families, `launch_fusion` and `norm`, now ship generated workspaces that can use Triton on real NVIDIA machines while still falling back cleanly to the reference path elsewhere.
  - Motivation: the next useful CUDA-lab proof after trace-first orchestration and deeper diagnosis was to show that the lab can host real backend-specific kernel code, not just plan for it. Starting with narrow residual-add and RMSNorm targets keeps the first Triton slice tractable without pretending the full CUDA starter catalog is already Triton-native.
  - Purpose: begin the actual Triton-backed CUDA workspace implementation while preserving the repo's current trace-first prioritization model and clean fallback behavior on non-CUDA developer machines.
  - Updated `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` so generated `launch_fusion` and `norm` workspaces include optional Triton kernels and annotate themselves as `triton-optional`.
  - Updated `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py` and the kernel-lab docs to make it explicit which CUDA starter targets are now genuinely Triton-backed and which remain reference-first.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_workspace.py autoresearch_cuda/lab_trace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target launch_fusion --workspace /tmp/cuda-triton-launch-fusion`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target norm --workspace /tmp/cuda-triton-norm`
  - `python3 -m py_compile /tmp/cuda-triton-launch-fusion/kernel.py /tmp/cuda-triton-norm/kernel.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-triton-launch-fusion --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-triton-norm --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this Apple machine, the generated Triton-backed workspaces still degrade cleanly through the existing `missing-runtime` path because PyTorch/CUDA are unavailable.
  - The first Triton slice is intentionally partial: only `launch_fusion` and `norm` are Triton-backed so far, while the rest of the CUDA starter catalog remains reference-first.

### March 12, 2026 — `36bea5b` — lab: make CUDA deep-profile evidence policy-driving — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Tightened the CUDA evidence loop so `deep-profile` changes ranking and promotion behavior instead of only adding another artifact to the ledger.
  - Meaning: a strong deeper diagnosis now raises confidence for starter-ready CUDA targets, while a weak or mixed diagnosis can block promotion and force manual CUDA review. The lab no longer treats all deep-profile results as equally helpful.
  - Motivation: once `deep-profile` existed, it was not enough to record the result and leave the rest of the workflow unchanged. A `compute-bound` or `bandwidth-bound` diagnosis should push a target forward more confidently than a weak `mixed` result, and a weak result should not quietly drift into promotion on timing share alone.
  - Purpose: make the CUDA trace-first workflow more trustworthy by ensuring deeper diagnosis has real consequences for orchestration and promotion instead of acting like optional decoration.
  - Updated `/Users/ent/Codex/autoresearch/autoresearch_lab/ledger.py` so strong deeper diagnoses increase evidence strength while weak or mixed ones reduce it.
  - Updated `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py` so `promotion-check` can now return `needs-manual-cuda-review` when a target has weak deeper diagnosis despite passing starter verification, and so starter-ready CUDA targets require `deep-profile` before moving toward trainer integration.
  - Updated the user-facing CUDA lab guidance to say explicitly that strong deeper diagnoses raise confidence while weak ones block promotion pending manual review.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `program.md`
  - `autoresearch_lab/ledger.py`
  - `autoresearch_cuda/lab_trace.py`
- Validation:
  - `python3 -m py_compile autoresearch_lab/ledger.py autoresearch_cuda/lab_trace.py`
  - `./.venv/bin/python - <<'PY'`
    `... synthetic ledger and deeper-diagnosis summary check ...`
    `PY`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/render_autonomy_badge.py`
- Measurements:
  - Strong deeper diagnoses now contribute more evidence bonus than generic trace-backed timing alone.
  - Weak or mixed deeper diagnoses now block starter-ready CUDA targets from drifting into trainer integration without manual review.

### March 12, 2026 — `f8779c0` — lab: add CUDA deeper kernel diagnosis — score `3` — complexity `8`

**AI-identified within brief, human-shaped (3)**

- Extended the CUDA trace-first lab with an optional deeper diagnosis stage built on Nsight Compute.
  - Meaning: CUDA trace review is no longer limited to Nsight Systems timing plus a broad automatic bottleneck label. A trace-ranked family can now be rerun through `deep-profile` so the lab records a more specific diagnosis such as `compute-bound`, `bandwidth-bound`, or `under-occupied`, and promotion/orchestration can use that evidence instead of treating all trace-backed targets the same.
  - Motivation: the CUDA trace path had reached the point where `capture`, `trace-profile`, and `auto-review` could tell us that a family mattered, but not why it mattered at the kernel level. Without that deeper signal, starter-ready CUDA families risked moving toward promotion on timing share alone.
  - Purpose: make CUDA the first backend where the lab can move beyond timing-only trace evidence and toward machine-generated kernel diagnosis that meaningfully shapes orchestration and promotion.
  - Added `LabDeepProfileResult` and deeper-diagnosis capability flags to the shared lab boundary in `/Users/ent/Codex/autoresearch/autoresearch_lab/labs.py`.
  - Extended the shared ledger in `/Users/ent/Codex/autoresearch/autoresearch_lab/ledger.py` to persist deeper CUDA diagnoses and let that evidence affect later promotion/ranking.
  - Added `deep-profile` to `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab.py` and implemented the Nsight Compute runner plus metric classification in `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py`.
  - Updated CUDA orchestration and promotion checks so starter-ready families like `norm`, `fused_mlp`, `matmul_epilogue`, and `rope_qk_fused` can explicitly require deeper diagnosis before moving further.
  - Updated the user-facing CUDA lab story so the trace-first path now includes `deep-profile` as the step between broad trace review and stronger promotion claims.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `program.md`
  - `autoresearch_lab/labs.py`
  - `autoresearch_lab/ledger.py`
  - `autoresearch_cuda/lab.py`
  - `autoresearch_cuda/lab_trace.py`
- Validation:
  - `python3 -m py_compile autoresearch_lab/labs.py autoresearch_lab/ledger.py autoresearch_cuda/lab.py autoresearch_cuda/lab_trace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda deep-profile --trace-profile /tmp/cuda-deep-profile.synthetic.json --rank 1`
  - `./.venv/bin/python - <<'PY'`
    `... synthetic Nsight Compute CSV parse/classification check for bandwidth-bound diagnosis ...`
    `PY`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/render_autonomy_badge.py`
- Measurements:
  - On this Apple machine, `deep-profile` degrades cleanly to `missing-tool` when `ncu` is unavailable instead of failing as an opaque shell error.
  - The first deeper diagnosis layer is intentionally narrow:
    - it uses three initial Nsight Compute metrics
    - it classifies `compute-bound`, `bandwidth-bound`, `under-occupied`, or `mixed`
    - and it records that result as structured evidence rather than treating it as a side note outside the lab loop.

### March 12, 2026 — `ba578ad` — lab: add CUDA RoPE and Q/K-normalization trainer hook — score `3` — complexity `8`

**AI-identified within brief, human-shaped (3)**

- Broadened the direct CUDA trainer-hook layer again by wiring the fused RoPE + Q/K normalization seam into both the starter workspace catalog and the real attention path.
  - Meaning: the CUDA lab can now exercise one more honest trainer seam immediately around the attention core:
    - RoPE application and Q/K RMSNorm via `rope_qk_fused`
      That fills the gap between `attention_prelude` and the attention kernel itself, so the CUDA starter-ready set now spans the practical narrow seams on both sides of the core attention op.
  - Motivation: after `attention_prelude`, the next natural narrow seam was the rotary-plus-Q/K-normalization path. It already exists as a repeated attention-side boundary in the trainer, is trace-visible, and is still much simpler than trying to hook the full attention core or FlashAttention path directly.
  - Purpose: complete the current narrow CUDA attention-side hook set before shifting the focus back toward stronger trace-backed evidence and broader promotion logic.
  - Added a `rope_qk_fused` starter workspace in `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with fixed quick/full harness cases and a fused reference path.
  - Extended `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_integration.py` so `rope_qk_fused` runs through the shared environment-driven workspace injection path.
  - Wired `/Users/ent/Codex/autoresearch/autoresearch_cuda/train.py` so the real Q/K rotary + normalization seam can call a `rope_qk_fused` workspace implementation before falling back to the existing separate rotary and norm path.
  - Updated `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py` so trace profiling can rank `rope_qk_fused` as its own starter-ready family instead of hiding that work under the broader `attention_prelude` bucket.
  - Updated the public CUDA lab story so all current starter-ready CUDA targets and direct trainer-side hooks now include `rope_qk_fused`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `program.md`
  - `autoresearch_cuda/lab_integration.py`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_integration.py autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py autoresearch_cuda/train.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target rope_qk_fused --workspace /tmp/cuda-rope-qk-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-rope-qk-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-rope-qk-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-rope-qk-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so the new `rope_qk_fused` integration commands degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The current starter-ready CUDA target set is now:
    - `launch_fusion`
    - `norm`
    - `loss_prelude`
    - `data_movement`
    - `matmul_epilogue`
    - `attention_prelude`
    - `rope_qk_fused`
    - `fused_mlp`
  - The current direct CUDA trainer-hook set is now identical to that starter-ready set, so every narrow CUDA starter target has a real trainer-side seam even though full validation still requires a CUDA-capable machine.

### March 12, 2026 — `0f29146` — lab: add residual-add and reshape CUDA trainer hooks — score `3` — complexity `8`

**AI-identified within brief, human-shaped (3)**

- Broadened the direct CUDA trainer-hook layer again by wiring the remaining practical generic workspace families into real trainer seams: residual-add launch fusion and the attention-output reshape path.
  - Meaning: the CUDA lab can now exercise two more repeated trainer-real paths without inventing fake integration stories:
    - residual-add launch fusion via `launch_fusion`
    - attention-output flattening via `data_movement`
      That pushes the direct CUDA integration set past only norms, loss-side prep, epilogues, and block-local compute into the small repeated glue seams that often show up as launch-bound or reshape-heavy work in traces.
  - Motivation: after `fused_mlp` and `attention_prelude`, the two remaining generic starter families still did not map onto honest trainer seams. The right next step was to retune those families around real residual-add and reshape boundaries instead of leaving them as synthetic-only workspaces.
  - Purpose: widen the CUDA direct-hook set with the last practical narrow seams before the next stage shifts from adding hooks toward stronger integration evidence and eventual CUDA workspace promotion.
  - Retuned `launch_fusion` in `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` so it models residual-add fusion, and retuned `data_movement` so it models the real attention-output reshape from `[B, T, H, D]` to `[B, T, H*D]`.
  - Extended `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_integration.py` so `launch_fusion` and `data_movement` both run through the shared environment-driven workspace injection path.
  - Wired `/Users/ent/Codex/autoresearch/autoresearch_cuda/train.py` so block residual adds can call a `launch_fusion` workspace implementation and the attention output reshape can call a `data_movement` workspace implementation before the existing projection path.
  - Updated the public CUDA lab story so all current starter-ready CUDA targets now have direct trainer-side hooks.
  - Kept the path narrow and honest: non-CUDA machines still return structured `missing-runtime` integration results instead of pretending the new seams were validated here.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `program.md`
  - `autoresearch_cuda/lab_integration.py`
  - `autoresearch_cuda/lab_workspace.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_workspace.py autoresearch_cuda/lab_integration.py autoresearch_cuda/train.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target launch_fusion --workspace /tmp/cuda-launch-fusion-integration-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-launch-fusion-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target data_movement --workspace /tmp/cuda-data-movement-integration-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-data-movement-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-data-movement-integration-workspace --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so the new `launch_fusion` and `data_movement` integration commands degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The direct CUDA trainer-hook set is now:
    - `launch_fusion`
    - `norm`
    - `loss_prelude`
    - `data_movement`
    - `matmul_epilogue`
    - `fused_mlp`
    - `attention_prelude`
  - That means every current starter-ready CUDA target now has a direct trainer-side seam, even though real trainer-side validation still requires a CUDA-capable machine.

### March 12, 2026 — `7d8699c` — lab: add broader CUDA block-path trainer hooks — score `3` — complexity `8`

**AI-identified within brief, human-shaped (3)**

- Broadened the direct CUDA trainer-hook layer from edge seams into block-local seams by adding `fused_mlp` and `attention_prelude`.
  - Meaning: the CUDA lab can now exercise two more trainer-real paths inside the transformer block itself:
    - the feed-forward path via `fused_mlp`
    - the Q/K/V staging path via `attention_prelude`
      That moves the direct CUDA integration story beyond normalization, loss-side prelude, and final projection epilogues into repeated block-local compute and attention setup regions.
  - Motivation: after the first narrow seams landed and the broader starter CUDA workspaces existed, the cleanest next direct hooks were the MLP path and the attention prelude. Both already had starter workspaces and map onto real trainer boundaries without forcing a fake full-attention or full-GEMM rewrite story.
  - Purpose: keep widening the direct CUDA trainer-hook set with seams that are still narrow enough to validate honestly, but broad enough to make promotion evidence about repeated block-local work rather than only edge fragments.
  - Extended `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_integration.py` so `fused_mlp` and `attention_prelude` both run through the shared environment-driven workspace injection path alongside the earlier direct targets.
  - Retuned the `attention_prelude` starter workspace in `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` so it now matches the real Q/K/V staging seam, including optional value-embed gating, instead of normalizing after projection inside the workspace.
  - Wired `/Users/ent/Codex/autoresearch/autoresearch_cuda/train.py` so the CUDA MLP path can call a `fused_mlp` workspace implementation and the attention path can call an `attention_prelude` workspace implementation before the existing rotary, norm, and attention-core logic.
  - Updated the public CUDA lab story so the direct trainer-hook set is now `norm`, `loss_prelude`, `matmul_epilogue`, `fused_mlp`, and `attention_prelude`.
  - Kept the path narrow and honest: other CUDA starter targets still stop at workspace-local evidence, and non-CUDA machines still return structured `missing-runtime` integration results.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `program.md`
  - `autoresearch_cuda/lab_integration.py`
  - `autoresearch_cuda/lab_workspace.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_workspace.py autoresearch_cuda/lab_integration.py autoresearch_cuda/train.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target fused_mlp --workspace /tmp/cuda-fused-mlp-integration-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-fused-mlp-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target fused_mlp --workspace /tmp/cuda-fused-mlp-integration-workspace-2`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-fused-mlp-integration-workspace-2 --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target attention_prelude --workspace /tmp/cuda-attention-prelude-integration-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-attention-prelude-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target attention_prelude --workspace /tmp/cuda-attention-prelude-integration-workspace-2`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-attention-prelude-integration-workspace-2 --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so the new `fused_mlp` and `attention_prelude` integration commands degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The direct CUDA trainer-hook set is now:
    - `norm`
    - `loss_prelude`
    - `matmul_epilogue`
    - `fused_mlp`
    - `attention_prelude`
  - Other CUDA starter targets still stop at workspace-local evidence until more trainer seams are wired.

### March 12, 2026 — `2c7d33f` — lab: add broader CUDA starter workspaces — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Promoted `attention_prelude` and `fused_mlp` from trace-only CUDA families into real starter workspaces with fixed harness support.
  - Meaning: the CUDA starter-workspace layer now covers two broader training-path families beyond launch fusion, norms, loss prelude, data movement, and matmul epilogues. Targets that previously stopped at trace-backed prioritization can now move through `extract`, `bench`, and `verify` like the other starter-ready CUDA families.
  - Motivation: after landing the first narrow trainer-hook seams, the next useful expansion was not another larger integration seam. It was giving the trace system broader starter-ready families so attention glue and MLP work can move into real workspaces without pretending they are already direct trainer hooks.
  - Purpose: broaden the practical CUDA workspace catalog while keeping the direct trainer-hook set narrow and honest, so the trace-first CUDA lab can keep expanding without overclaiming promotion readiness.
  - Added `attention_prelude` and `fused_mlp` starter templates plus fixed benchmark cases to `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py`.
  - Promoted both families to `starter-ready` in `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py`, so orchestration can emit real `extract` / `bench` / `verify` plans for them.
  - Updated the public CUDA lab story so the starter-ready family list is broader, while the direct trainer-hook set remains `norm`, `loss_prelude`, and `matmul_epilogue`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab.py autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda list-targets`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target attention_prelude --workspace /tmp/cuda-attention-prelude-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-attention-prelude-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-attention-prelude-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target fused_mlp --workspace /tmp/cuda-fused-mlp-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-fused-mlp-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-fused-mlp-workspace --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so the new `attention_prelude` and `fused_mlp` bench/verify commands degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The CUDA starter-ready workspace set now includes:
    - `launch_fusion`
    - `norm`
    - `loss_prelude`
    - `data_movement`
    - `matmul_epilogue`
    - `attention_prelude`
    - `fused_mlp`
  - The direct CUDA trainer-hook set intentionally remains narrower:
    - `norm`
    - `loss_prelude`
    - `matmul_epilogue`
  - Broader CUDA trace families still remain outside the starter-workspace set until more fixed harnesses are added.

### March 12, 2026 — `2e4cc37` — lab: broaden CUDA trainer-side integration seams — score `3` — complexity `7`

**AI-identified within brief, human-shaped (3)**

- Broadened the direct CUDA trainer-hook layer from `norm` and `loss_prelude` to also include `matmul_epilogue`.
  - Meaning: the CUDA lab can now exercise three distinct narrow trainer seams:
    - normalization
    - loss-side prelude
    - final projection epilogue
      That gives the CUDA integration story coverage on more than one part of the training stack before we attempt any broader family.
  - Motivation: after `norm` and `loss_prelude`, the cleanest next seam was the final projection. `matmul_epilogue` already existed as a starter workspace, and the trainer has a clear final-projection boundary where a narrow epilogue hook can be inserted without committing to a full GEMM rewrite story.
  - Purpose: keep widening the CUDA trainer-hook layer with seams that are both narrow and trainer-real, so future promotion evidence can move beyond one-off path fragments and start comparing different classes of kernel work under the same lab loop.
  - Extended `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_integration.py` so `matmul_epilogue` can run through the shared environment-driven workspace injection path alongside `norm` and `loss_prelude`.
  - Wired `/Users/ent/Codex/autoresearch/autoresearch_cuda/train.py` so the final projection can call a `matmul_epilogue` workspace implementation before the logits-softcap path, with a zero-bias fallback for the starter workspace contract.
  - Updated the public CUDA lab story so the direct trainer-hook set is now `norm`, `loss_prelude`, and `matmul_epilogue`.
  - Kept the path narrow and honest: broader CUDA starter targets still stop at workspace-local evidence, and non-CUDA machines still return structured `missing-runtime` integration results.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `program.md`
  - `autoresearch_cuda/lab_integration.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_integration.py autoresearch_cuda/train.py autoresearch_cuda/lab.py autoresearch_cuda/lab_trace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target matmul_epilogue --workspace /tmp/cuda-matmul-integration-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-matmul-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-matmul-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so the new `matmul_epilogue` integration commands degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The direct CUDA trainer-hook set is now intentionally narrow but no longer small enough to be one-path-specific:
    - `norm`
    - `loss_prelude`
    - `matmul_epilogue`
  - Broader CUDA starter targets still stop at workspace-local evidence until more trainer seams are wired.

### March 12, 2026 — `ff37e76` — lab: expand CUDA direct trainer hooks — score `3` — complexity `7`

**AI-identified within brief, human-shaped (3)**

- Expanded the narrow CUDA trainer-hook layer from `norm` alone to `norm` plus `loss_prelude`.
  - Meaning: the direct CUDA trainer-side integration path now covers both a normalization seam and a loss-side seam, so the CUDA lab can start collecting trainer evidence from more than one narrow target family instead of treating `norm` as the only special case.
  - Motivation: once the first CUDA trainer hook landed, the cleanest next step was another target that already existed as a starter workspace and matched a clear trainer seam. The logits-softcap-plus-loss prelude path is the narrowest useful follow-on before broader families like data movement or epilogues.
  - Purpose: widen the CUDA trainer-hook layer carefully, keep the path tied to existing starter workspaces, and keep non-CUDA machines on the same structured `missing-runtime` workflow instead of hand-maintained special cases.
  - Extended `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_integration.py` so `loss_prelude` can run through the same environment-driven workspace injection path as `norm`.
  - Wired `/Users/ent/Codex/autoresearch/autoresearch_cuda/train.py` so the logits softcap and training-loss seam can call a `loss_prelude` workspace implementation safely, including masked `ignore_index=-1` handling.
  - Updated the public CUDA lab story so the direct trainer-hook set is now explicitly `norm` plus `loss_prelude`.
  - Kept the path narrow and honest: broader CUDA starter targets still stop at workspace-local evidence, and non-CUDA machines still return structured `missing-runtime` integration results.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `program.md`
  - `autoresearch_cuda/lab_integration.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab_integration.py autoresearch_cuda/train.py autoresearch_cuda/lab.py autoresearch_cuda/lab_trace.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target loss_prelude --workspace /tmp/cuda-loss-integration-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-loss-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-loss-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so both `loss_prelude` integration commands degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The direct CUDA trainer-hook set is now intentionally narrow but no longer single-target:
    - `norm`
    - `loss_prelude`
  - Broader CUDA starter targets still stop at workspace-local evidence until more trainer seams are wired.

### March 12, 2026 — `89bbc11` — lab: add first CUDA trainer hook and integration evidence — score `3` — complexity `7`

**AI-identified within brief, human-shaped (3)**

- Added the first direct CUDA trainer hook and trainer-side integration evidence path for a starter workspace target.
  - Meaning: CUDA kernel-lab now has the first path that can move a target from workspace-local evidence into real trainer-side A/B evidence, instead of stopping at trace-backed workspace work. The first hooked target is `norm`.
  - Motivation: once the CUDA lab had trace-first orchestration plus starter workspaces, the next missing step was the same one MLX had to solve earlier: a way to prove that a narrow starter kernel still matters in the live trainer rather than only in its fixed harness.
  - Purpose: start the CUDA trainer-integration layer one narrow target at a time, keep the path conservative, and let the lab record structured `missing-runtime` outcomes on non-CUDA machines instead of pretending the integration story is usable everywhere already.
  - Added `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_integration.py` with a first direct integration target set and environment wiring for trainer-side A/B runs.
  - Wired `/Users/ent/Codex/autoresearch/autoresearch_cuda/train.py` so the shared `norm(...)` path can call a lab workspace implementation when the CUDA lab integration environment is active.
  - Added `integration-ab` and `integration-suite` to `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab.py`, with orchestration, promotion, and evidence flowing through `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_trace.py`.
  - Kept the path narrow and honest: only `norm` is directly integrable today, and non-CUDA machines receive structured `missing-runtime` results instead of fake promotion progress.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `program.md`
  - `autoresearch_cuda/lab.py`
  - `autoresearch_cuda/lab_integration.py`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab.py autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py autoresearch_cuda/lab_integration.py autoresearch_cuda/train.py autoresearch_lab/labs.py autoresearch_lab/entrypoints.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target norm --workspace /tmp/cuda-norm-integration-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-norm-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-norm-integration-workspace --preset upstream --time-budget 2 --repeats 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine cuda promotion-check --target norm --preset upstream`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so both `integration-ab` and `integration-suite` degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The first direct CUDA trainer hook is intentionally narrow:
    - `norm` is directly integrable
    - broader CUDA starter targets still stop at workspace-local evidence
  - `promotion-check` for `norm` remains conservative and still reports `needs-capture` until trace-backed evidence exists.

### March 12, 2026 — `e68136a` — lab: add more CUDA starter workspace families — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Promoted `data_movement` and `matmul_epilogue` from trace-only CUDA families into real starter workspaces with fixed harness support.
  - Meaning: the CUDA starter-workspace layer now covers two more trace-visible families beyond pointwise fusion, norms, and loss-prelude work. Targets that previously stopped at `ready-for-cuda-workspace-family` can now move through `extract`, `bench`, and `verify` like the other narrow CUDA starters.
  - Motivation: once the first CUDA starter set landed, `data_movement` was the cleanest next family to promote and `matmul_epilogue` was the next most practical step toward more trainer-relevant CUDA work without jumping directly into full attention kernels.
  - Purpose: broaden the CUDA workspace layer one family at a time while keeping the trace-first architecture intact and the harnesses honest.
  - Added `data_movement` and `matmul_epilogue` starter templates plus fixed benchmark cases to `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py`.
  - Promoted both targets to `starter-ready` in the CUDA trace catalog so orchestration and promotion-check treat them like real workspace-capable families.
  - Updated the docs to list both families alongside the other current CUDA starter workspaces.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab.py autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py autoresearch_lab/labs.py autoresearch_lab/entrypoints.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda list-targets`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target data_movement --workspace /tmp/cuda-data-movement-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-data-movement-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-data-movement-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target matmul_epilogue --workspace /tmp/cuda-matmul-epilogue-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-matmul-epilogue-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-matmul-epilogue-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda extract --profile /tmp/cuda-trace-sim.profile.json --workspace /tmp/cuda-data-movement-extract --rank 2`
  - `./.venv/bin/python kernel-lab.py --engine cuda promotion-check --target data_movement --preset upstream`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so `data_movement` and `matmul_epilogue` bench/verify degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The synthetic trace profile now lets `extract --rank 2` instantiate a real `data_movement` workspace instead of returning `starter-unavailable`.
  - `promotion-check` for `data_movement` now reports `ready-for-cuda-workspace` instead of `ready-for-cuda-workspace-family`.

### March 12, 2026 — `59c4136` — lab: add CUDA starter workspaces and fixed harnesses — score `4` — complexity `9`

**Human-directed, AI-shaped (4)**

- Added the first narrow CUDA workspace layer on top of the trace-first lab, covering starter-ready `launch_fusion`, `norm`, and `loss_prelude` families with `init`, `bench`, `verify`, and `extract`.
  - Meaning: CUDA kernel-lab is no longer only “capture, classify, and plan.” For a few well-chosen families, the repo can now open a mutable workspace, run a fixed harness, and record workspace-local evidence under the same shared lab boundary.
  - Motivation: the trace-first path was the right first move on NVIDIA, but it still stopped at “this looks important.” The next useful proof is that a trace-backed CUDA target can turn into a real workspace loop without forcing the whole backend to wait for a full Triton integration story.
  - Purpose: connect automated CUDA trace review to actual workspace work, while keeping scope narrow enough that the first CUDA workspace layer is trustworthy and easy to extend.
  - Added `/Users/ent/Codex/autoresearch/autoresearch_cuda/lab_workspace.py` with starter templates, fixed benchmark cases, and correctness harnesses for the first three CUDA target families.
  - Updated the CUDA lab capabilities and CLI so `kernel-lab.py --engine cuda` now supports `init`, `bench`, `verify`, and `extract` in addition to the existing trace-first commands.
  - Made CUDA orchestration emit real `extract` / `bench` / `verify` commands for starter-ready targets, while keeping broader families trace-backed planning targets.
  - Tightened CUDA promotion states so starter-ready families and broader future workspace families are no longer described the same way.
  - Updated the docs to explain the new CUDA split: trace-first by default, with a small starter-workspace layer where the target family is narrow enough to benchmark honestly.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_cuda/lab.py`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_cuda/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_cuda/lab.py autoresearch_cuda/lab_trace.py autoresearch_cuda/lab_workspace.py autoresearch_lab/labs.py autoresearch_lab/entrypoints.py`
  - `./.venv/bin/python kernel-lab.py --engine cuda list-targets`
  - `./.venv/bin/python kernel-lab.py --engine cuda init --target launch_fusion --workspace /tmp/cuda-launch-fusion-workspace`
  - `./.venv/bin/python kernel-lab.py --engine cuda bench --workspace /tmp/cuda-launch-fusion-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda verify --workspace /tmp/cuda-launch-fusion-workspace --quick`
  - `./.venv/bin/python kernel-lab.py --engine cuda extract --profile /tmp/cuda-trace-sim.profile.json --workspace /tmp/cuda-loss-workspace --rank 2`
  - `./.venv/bin/python kernel-lab.py --engine cuda orchestrate --trace-profile /tmp/cuda-trace-sim.profile.json --workspace-root /tmp/cuda-kernel-orch2 --rank 1`
  - `./.venv/bin/python kernel-lab.py --engine cuda evidence --target launch_fusion --preset upstream`
  - `./.venv/bin/python kernel-lab.py --engine cuda promotion-check --target launch_fusion --preset upstream`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `torch` is not installed, so CUDA `bench` and `verify` now degrade cleanly to `missing-runtime` instead of failing as opaque import errors.
  - The synthetic launch-bound trace-profile fixture still drives the trace-first half:
    - `orchestrate -> trace-prioritized`
    - `evidence -> trace-backed`
  - For starter-ready families, `promotion-check` now reports `ready-for-cuda-workspace`.
  - For broader families like `data_movement`, `promotion-check` now reports `ready-for-cuda-workspace-family`, making the distinction between “workspace exists” and “workspace family should be added” explicit.

### March 12, 2026 — `a3b806e` — lab: add CUDA trace evidence and orchestration — score `4` — complexity `10`

**Human-directed, AI-shaped (4)**

- Added the first CUDA trace-first kernel-lab path through Nsight capture, machine-readable trace profiling, automated first-pass review, and trace-backed orchestration, without pretending CUDA workspaces exist yet.
  - Meaning: the shared kernel-lab boundary now covers a second backend in a substantively different way. MLX remains the deep workspace-first path, while CUDA starts with trace-first commands (`capture`, `trace-profile`, `auto-review`, `evidence`, `orchestrate`, `promotion-check`) because that is where the NVIDIA tooling advantage actually is.
  - Motivation: MLX proved the kernel-lab workflow, but Apple trace truth is still GUI/manual. CUDA is the backend where the repo can start automating trace review for real, so the next useful proof is scripted trace capture, classification, evidence tracking, and next-step selection rather than rushing straight into Triton workspaces.
  - Purpose: make real CUDA trace evidence a first-class lab artifact, establish the cross-engine boundary on something operationally meaningful, and ensure later CUDA/Triton workspace work starts from measured bottlenecks instead of guesses.
  - Extended the shared lab boundary with trace-profile / auto-review result types and capability flags, and registered `--engine cuda` at the top-level `kernel-lab.py` front door.
  - Added a CUDA target-family catalog plus a new trace-first lab implementation that can capture trainer runs under Nsight Systems, store `.nsys-rep` metadata sidecars, and summarize exported reports into ranked kernel target families.
  - Added `trace-profile` and `auto-review` commands so CUDA can classify end-to-end runs as launch-bound, sync-bound, copy-bound, kernel-dominated, or mixed, and record that result into the shared ledger as machine-generated evidence.
  - Added CUDA `evidence` and `promotion-check` so trace-backed targets can now be summarized and gated through the same shared ledger model as MLX, even before CUDA workspaces exist.
  - Added CUDA `orchestrate` so a trace-profile artifact plus accumulated evidence can now answer “what should we optimize next?” instead of stopping at a passive report.
  - Updated the user and agent docs to explain the new split: MLX capture remains human-reviewed, while CUDA is now the first backend where the repo aims for automated trace review by default.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_lab/entrypoints.py`
  - `autoresearch_lab/labs.py`
  - `autoresearch_cuda/lab.py`
  - `autoresearch_cuda/lab_trace.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_lab/labs.py autoresearch_lab/entrypoints.py autoresearch_cuda/lab.py autoresearch_cuda/lab_trace.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --list-engines`
  - `./.venv/bin/python kernel-lab.py --engine cuda list-targets`
  - `./.venv/bin/python kernel-lab.py --engine cuda capture --preset upstream --time-budget 1 --output /tmp/cuda-trace-foundation`
  - `./.venv/bin/python kernel-lab.py --engine cuda trace-profile --metadata /tmp/cuda-trace-foundation.metadata.json --output /tmp/cuda-trace-foundation.profile.json`
  - `./.venv/bin/python kernel-lab.py --engine cuda auto-review --trace-profile /tmp/cuda-trace-foundation.profile.json`
  - `python3 - <<'PY' ...` to write `/tmp/cuda-trace-sim.profile.json` with a synthetic launch-bound trace-profile fixture
  - `./.venv/bin/python kernel-lab.py --engine cuda orchestrate --trace-profile /tmp/cuda-trace-sim.profile.json --workspace-root /tmp/cuda-kernel-orch --rank 1`
  - `./.venv/bin/python kernel-lab.py --engine cuda evidence --target launch_fusion --preset upstream`
  - `./.venv/bin/python kernel-lab.py --engine cuda promotion-check --target launch_fusion --preset upstream`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - On this machine `nsys` is not installed, so `capture` returns a structured `missing-tool` result instead of crashing.
  - The downstream CUDA commands degrade cleanly from that state:
    - `trace-profile -> capture-unavailable`
    - `auto-review -> insufficient-trace-data`
  - A synthetic launch-bound trace-profile fixture was enough to exercise the post-capture path:
    - `orchestrate -> trace-prioritized`
    - `evidence -> trace-backed`
    - `promotion-check -> ready-for-cuda-workspace`
  - The top-level engine list now includes both `mlx` and `cuda`.

### March 12, 2026 — `4be493b` — train: rename the M5 preset ladder and retune shipped defaults — score `4` — complexity `9`

**Human-directed, AI-shaped (4)**

- Renamed the M5 preset ladder so the names scale around the current best validation-centered local target, then retuned the shipped defaults and refreshed the docs/table to match the measured operating points.
  - Meaning: the ladder is now `m5-tiny`, `m5-small`, `m5-balanced`, `m5-large`, `m5-xlarge`, where `m5-balanced` names the current best validation-centered local target, `m5-small` stays the productive default start, and `m5-large` is the 50.3M upstream-leaning bridge preset.
  - Motivation: the old names mixed together “best starting point,” “best validation target,” and “bigger model” in a way that no longer matched the measured M5 results. The README table also still had stale batch settings and missing last-loss values.
  - Purpose: make the preset scale easier to reason about, keep the shipped runtime defaults aligned with the validated M5 operating points, and make the beginner-facing story line up with the real “M5 laptop ↔ H100-shaped upstream” spectrum that `calibrate.py` is meant to explain.
  - Updated the MLX preset registry and platform preset ordering around the renamed ladder, removed the old preset-name shims from live code, and migrated the active checkpoint/default/telemetry/ledger artifacts in place so the runtime only sees canonical names.
  - Restamped the seeded M5 eval-calibration rows onto the renamed ladder and kept the bridge preset on the explicit `missing-calibration` fallback path until it gets its own checked-in eval row.
  - Refreshed the bring-up M5 reference table so platform calibration compares against the renamed ladder and newer batch/window defaults rather than stale lower-batch numbers.
  - Reworked the README preset section and examples around the new ladder semantics and filled in the missing 5-minute last-loss values in the reference table.
  - Updated the MLX agent supplement and preset-calibration docs so they describe the renamed ladder and the new seeded-vs-unseeded calibration coverage correctly.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `docs/preset-calibration.md`
  - `docs/program-mlx.md`
  - `autoresearch_mlx/eval_policy.py`
  - `autoresearch_mlx/lab_workspace.py`
  - `autoresearch_mlx/train.py`
  - `autoresearch_platform/mlx_engine.py`
  - `tools/calibrate_eval_policy.py`
  - `tools/calibrate_platform.py`
  - `tools/profile_loader_path.py`
  - `tools/profile_checkpoint_path.py`
  - `tools/profile_resume_convergence.py`
  - `tools/profile_resume_ready.py`
- Validation:
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/render_autonomy_badge.py`
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/eval_policy.py autoresearch_mlx/lab_workspace.py autoresearch_platform/mlx_engine.py tools/calibrate_eval_policy.py tools/calibrate_platform.py tools/profile_loader_path.py tools/profile_checkpoint_path.py tools/profile_resume_convergence.py tools/profile_resume_ready.py`
  - `./.venv/bin/python train.py --engine mlx --preset m5-fast --time-budget 0.05 --no-checkpoint` (rejected old preset name as expected)
  - `./.venv/bin/python train.py --engine mlx --preset m5-small --time-budget 0.05 --no-checkpoint`
  - `./.venv/bin/python calibrate.py --engine mlx --mode fast --coarse-time-budget 0.2 --ranking-time-budget 0.2 --local-search-time-budget 0.2 --eval-train-seconds 0.2 --eval-rungs cheap,reference --output-dir /tmp/autoresearch_preset_rename_fast2 --force`
  - `./.venv/bin/python kernel-lab.py --engine mlx promotion-check --target logits_softcap --workspace /tmp/mlx-logits-integration`
- Measurements:
  - Old preset names are no longer accepted in the live trainer path.
  - A fresh `m5-small` run now resolves `eval_calibration_status=calibrated` again with matching current signatures.
  - A fresh fast bring-up rewrote the cached MLX platform default to canonical names and current signatures:
    - preset: `m5-tiny`
    - `eval_semantics_signature=896401b41fddf1c4`
    - `runtime_shape_signature=2706200255536d41`
  - No-preset kernel-lab promotion checks now resolve directly from that refreshed calibrated platform default with no preset-translation layer.

### March 12, 2026 — `2a86f0d` — lab: strengthen MLX integration evidence with repeated and cross-preset A/Bs — score `3` — complexity `7`

**AI-identified within brief, human-shaped (3)**

- Strengthened MLX kernel-lab integration evidence so promotion depends on repeated balanced trainer A/B runs and cross-preset coverage instead of treating a single encouraging run as nearly sufficient.
  - Meaning: `integration-ab` now runs repeated measured rounds in alternating order after warmup, aggregates median trainer summaries, records pair count plus relative deltas, and only lets the ledger call a target `integration-validated` or `integration-regressed` when the evidence is both repeated and directionally consistent.
  - Motivation: the old trainer-side bridge closed the loop, but it still overfit to one-off runs. Kernel promotion needed stronger evidence than a single baseline/candidate pair because short local runs are noisy enough to flip sign.
  - Purpose: make `promotion-check`, orchestration, and later trainer patch promotion depend on repeated end-to-end evidence that is harder to fool with compile noise or transient drift, and push stronger candidates onto at least one heavier operating point before promotion.
  - Upgraded `integration-ab` to repeated balanced rounds and stored richer aggregate details, including relative throughput deltas and measured pair count.
  - Tightened ledger aggregation so `integration-validated` and `integration-regressed` require stronger repeated evidence, while weak or mixed results stay in `integration-tested` or `integration-mixed`.
  - Added `integration-suite`, which defaults to the calibrated point plus the next stronger preset, and feeds that broader evidence back into `promotion-check`.
  - Updated orchestration and promotion messaging to point at repeated A/B reruns on the calibrated point and a stronger preset rather than implying that one run is close to promotion.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `autoresearch_lab/labs.py`
  - `autoresearch_lab/ledger.py`
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_profile.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_lab/labs.py autoresearch_lab/ledger.py autoresearch_mlx/lab.py autoresearch_mlx/lab_profile.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx integration-ab --workspace /tmp/mlx-logits-integration --time-budget 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine mlx integration-suite --workspace /tmp/mlx-logits-integration --time-budget 1 --repeats 1 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine mlx evidence --target logits_softcap --preset m5-fast`
  - `./.venv/bin/python kernel-lab.py --engine mlx evidence --target logits_softcap`
  - `./.venv/bin/python kernel-lab.py --engine mlx promotion-check --target logits_softcap --workspace /tmp/mlx-logits-integration`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - A repeated balanced `integration-ab` run on `logits_softcap` completed at the calibrated platform default (`m5-fast`) and added `integration_ab_pair_count=4`.
  - The new `integration-suite` run added a second operating point (`m5-balanced`), bringing the overall evidence to:
    - `integration_ab_pair_count=8`
    - `integration_ab_preset_count=2`
    - `integration_ab_presets=["m5-balanced", "m5-fast"]`
  - The target still remains `integration-mixed`, with:
    - `integration_ab_positive_count=2`
    - `integration_ab_negative_count=2`
    - preset-local `integration_ab_median_delta_steady_state_tok_per_sec=-3409.9` on `m5-fast`
    - overall `integration_ab_median_delta_steady_state_tok_per_sec=-2836.5`
    - `integration_ab_median_delta_relative_pct_steady_state_tok_per_sec=null` because the older one-off events did not carry relative deltas, so the stricter ledger now refuses to infer a mixed absolute/relative aggregate.
  - `promotion-check` now uses the broader cross-preset evidence by default when it exists and keeps mixed targets in the stabilization path instead of treating one positive run as enough for promotion.

### March 12, 2026 — `335d608` — lab: default MLX integration checks to calibrated presets — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Made the MLX lab default its real trainer-side checks to the calibrated platform default for the current device, so promotion and integration tests exercise the machine’s actual recommended starting point instead of relying on hand-pinned presets.
  - Meaning: beyond smoke checks, `promotion-check` and `integration-ab` now behave like the rest of the platform story. They can discover the calibrated default for the current hardware, use it automatically, and reject stale cached defaults whose signatures no longer match the current code.
  - Motivation: once trainer-side integration A/B existed, the remaining gap was practical consistency. Real kernel promotion work should usually run against the device’s calibrated point, not against whichever preset happened to be typed into a command.
  - Purpose: keep kernel-lab integration evidence aligned with the same machine-specific default-selection flow used by `calibrate.py`, while still allowing explicit preset overrides for smoke or deliberately targeted tests.
  - Added a shared platform-default cache and write path so `calibrate.py` persists the latest candidate default for `engine x hardware_key`.
  - Made MLX `promotion-check` and `integration-ab` resolve the current calibrated default automatically when `--preset` is omitted, with clear fallback errors when no cache exists or the cached signatures are stale.
  - Updated the docs so the lab workflow now presents manual preset pinning as the exception rather than the normal path.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `docs/platform-calibration.md`
  - `docs/preset-calibration.md`
  - `autoresearch_lab/labs.py`
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_workspace.py`
  - `autoresearch_platform/platform_defaults.py`
  - `tools/calibrate_platform.py`
- Validation:
  - `python3 -m py_compile autoresearch_platform/platform_defaults.py autoresearch_lab/labs.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py tools/calibrate_platform.py`
  - `./.venv/bin/python calibrate.py --engine mlx --mode fast --coarse-time-budget 0.5 --ranking-time-budget 0.5 --local-search-time-budget 0.5 --eval-train-seconds 0.5 --eval-rungs cheap,reference --output-dir /tmp/autoresearch_mlx_postfix_fast`
  - `./.venv/bin/python kernel-lab.py --engine mlx promotion-check --target logits_softcap --workspace /tmp/mlx-logits-integration`
  - `./.venv/bin/python kernel-lab.py --engine mlx integration-ab --workspace /tmp/mlx-logits-integration --time-budget 2 --benchmark-skip-eval --no-checkpoint`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - A fresh MLX bring-up now writes `/Users/ent/.cache/autoresearch/platform_defaults/mlx/apple-m5-32gb-10gpu.json`.
  - With no `--preset`, `promotion-check` resolved `preset=m5-fast` with `preset_source=calibrated-platform-default`.
  - A no-preset `integration-ab` run against `logits_softcap` was recorded against the calibrated point and kept the target in `integration-mixed`, which is the intended conservative behavior.

### March 12, 2026 — `89df91f` — lab: Add MLX trainer integration A/B workflow — score `4` — complexity `8`

**Human-directed, AI-shaped (4)**

- Added the first real trainer-side MLX integration path for kernel-lab targets that already have narrow direct hooks, so promotion can advance from “trace-backed workspace” to “measured end-to-end baseline vs candidate run” instead of stopping at instructions.
  - Meaning: the lab now has a direct bridge into the real MLX trainer for a subset of targets. It can run a warmup + measured A/B, record the result in the ledger, and distinguish targets that are merely trace-backed from targets that have actually been tested in the live training path.
  - Motivation: after trace-backed orchestration and promotion checks landed, the remaining gap was the most important one: promotion-ready targets still did not have a built-in way to prove they helped the real trainer. The loop needed to stop emitting “next, run an integration A/B” as a manual idea and start doing it.
  - Purpose: close the lab loop so directly integrated targets can move from microbench and trace evidence into trainer evidence, while broader composed targets remain clearly marked as needing integration adapters before that comparison is meaningful.
  - Added `autoresearch_mlx/lab_integration.py` plus trainer hooks in `autoresearch_mlx/model.py` for a first set of directly integrable MLX targets.
  - Added `kernel-lab.py --engine mlx integration-ab ...`, which runs a fair warmup + measured baseline-vs-candidate trainer comparison and records the result.
  - Taught the evidence ledger to aggregate integration A/B outcomes across runs instead of trusting only the latest result, including `integration-validated`, `integration-regressed`, and `integration-mixed`.
  - Updated orchestration and promotion checks so direct targets can graduate into integration work, while broader targets such as `block_prelude` explicitly surface as `needs-integration-adapter`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_lab/labs.py`
  - `autoresearch_lab/ledger.py`
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_integration.py`
  - `autoresearch_mlx/lab_profile.py`
  - `autoresearch_mlx/lab_workspace.py`
  - `autoresearch_mlx/model.py`
- Validation:
  - `python3 -m py_compile autoresearch_lab/labs.py autoresearch_lab/ledger.py autoresearch_mlx/lab.py autoresearch_mlx/lab_profile.py autoresearch_mlx/lab_workspace.py autoresearch_mlx/model.py autoresearch_mlx/lab_integration.py kernel-lab.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target logits_softcap --workspace /tmp/mlx-logits-integration`
  - `./.venv/bin/python kernel-lab.py --engine mlx integration-ab --workspace /tmp/mlx-logits-integration --preset m5-fast --time-budget 2 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python kernel-lab.py --engine mlx evidence --target logits_softcap --preset m5-fast`
  - `./.venv/bin/python kernel-lab.py --engine mlx promotion-check --target logits_softcap --preset m5-fast --workspace /tmp/mlx-logits-integration`
  - `./.venv/bin/python kernel-lab.py --engine mlx promotion-check --target block_prelude --preset m5-balanced --workspace /tmp/mlx-kernel-workspace-2`
  - `./.venv/bin/python kernel-lab.py --engine mlx profile --preset m5-fast --top-k 12`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - `logits_softcap` now supports real trainer-side A/B runs through `integration-ab`.
  - Two measured `m5-fast` A/B runs produced mixed evidence for `logits_softcap`:
    - run 1: `steady_state_tok_per_sec_delta=+845.1`
    - run 2: `steady_state_tok_per_sec_delta=-4674.1`
    - summary: `integration_ab_ok_count=2`, `integration_ab_median_delta_steady_state_tok_per_sec=-1914.5`, `promotion_status=integration-mixed`
  - `promotion-check` now reports:
    - `logits_softcap -> integration-mixed`
    - `block_prelude -> needs-integration-adapter`

### March 12, 2026 — `00eecef` — lab: Add trace reviews, evidence summaries, and promotion checks — score `4` — complexity `8`

**Human-directed, AI-shaped (4)**

- Extended the MLX kernel-lab loop above raw verify/capture by adding explicit evidence summaries, trace-review signals, and promotion checks, so the system can decide not just "what next target should I try?" but also "is this target ready for an end-to-end A/B?" and "did the trace actually make it less important?"
  - Meaning: the lab now has a real middle policy layer above the ledger. It can summarize current evidence for a target, expose promotion readiness directly, and record whether a human or agent judged a trace as high- or low-value after opening it in Xcode.
  - Motivation: after persistent evidence and promotion-ready orchestration landed, the next gaps were practical ones: there was no direct way to inspect that evidence, no explicit promotion gate for end-to-end A/B, and no way for trace review to lower a target's priority when the backend reality looked weaker than the heuristic expected.
  - Purpose: turn the kernel lab into a more complete optimization loop where evidence can both promote and demote targets, and where the next action is explicit instead of buried inside a generic orchestration plan.
  - Added `evidence` and `promotion-check` commands to expose the current MLX ledger state and integration readiness directly.
  - Added `review-trace` so human/agent trace interpretation can be recorded as `none|low|medium|high` instead of disappearing in an external Xcode session.
  - Made evidence summaries trace-review-aware, including negative signals that can temporarily deprioritize a target.
  - Updated MLX profile ranking and promotion checks to use that richer evidence model instead of just "has verify" / "has capture".

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_lab/labs.py`
  - `autoresearch_lab/ledger.py`
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_profile.py`
  - `autoresearch_mlx/lab_trace.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_lab/labs.py autoresearch_lab/ledger.py autoresearch_mlx/lab.py autoresearch_mlx/lab_profile.py autoresearch_mlx/lab_trace.py autoresearch_mlx/lab_workspace.py kernel-lab.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx evidence --target block_prelude --preset m5-balanced`
  - `./.venv/bin/python kernel-lab.py --engine mlx promotion-check --target block_prelude --preset m5-balanced`
  - `./.venv/bin/python kernel-lab.py --engine mlx review-trace --workspace /tmp/mlx-kernel-workspace-2 --metadata /tmp/mlx-kernel-workspace-2/block-prelude-trace.metadata.json --relevance low --notes 'validation: weak end-to-end impact check'`
  - `./.venv/bin/python kernel-lab.py --engine mlx profile --preset m5-balanced --top-k 3`
  - `./.venv/bin/python kernel-lab.py --engine mlx review-trace --workspace /tmp/mlx-kernel-workspace-2 --metadata /tmp/mlx-kernel-workspace-2/block-prelude-trace.metadata.json --relevance high --notes 'validation: restore promotion-ready state after negative-signal check'`
  - `./.venv/bin/python kernel-lab.py --engine mlx profile --preset m5-balanced --top-k 2`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - `promotion-check` now reports `block_prelude` as `ready-for-integration-ab` when verify and capture evidence are present.
  - A recorded `low` trace review demoted `block_prelude` from rank `1` to rank `2` for `m5-balanced`, with:
    - `effective_priority_score=7.775`
    - `promotion_status=trace-deprioritized`
  - A later recorded `high` trace review restored it to rank `1`, with:
    - `effective_priority_score=9.325`
    - `promotion_status=ready-for-integration-test`

### March 12, 2026 — `587d4a8` — lab: Add evidence-aware MLX ranking and promotion — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Added a persistent MLX kernel-lab evidence ledger and made both ranking and orchestration react to it, so the next step can advance from fresh exploration to trace-backed promotion instead of redoing the same workspace loop forever.
  - Meaning: the kernel lab now has a third layer between heuristics and traces: a ledger that remembers successful `verify` and `capture` events by target/preset/backend, and uses that memory to adjust candidate ranking and pick a different next workflow when a target is already proven enough.
  - Motivation: after trace-backed orchestration landed, the next gap was that the system still forgot everything between commands. The lab needed a persistent notion of "already verified", "already traced", and "ready for integration test" so orchestration could stop treating every target as brand new.
  - Purpose: turn the kernel lab from a task suggester into the beginning of a real optimization loop, where evidence accumulates over time and promotion decisions become explicit rather than informal.
  - Added `autoresearch_lab/ledger.py` with persistent JSONL events and evidence summaries.
  - Wired MLX `verify` and `capture` to append ledger events, including recapture to an existing `.gputrace` path.
  - Made MLX profile ranking evidence-aware and taught orchestration to reuse existing workspaces, auto-load trace metadata when available, and emit `promotion-ready` plans when both verify and trace evidence exist.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_lab/ledger.py`
  - `autoresearch_mlx/lab_profile.py`
  - `autoresearch_mlx/lab_trace.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile autoresearch_lab/ledger.py autoresearch_mlx/lab_trace.py autoresearch_mlx/lab_profile.py autoresearch_mlx/lab_workspace.py autoresearch_mlx/lab.py kernel-lab.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx verify --workspace /tmp/mlx-kernel-workspace-2 --quick`
  - `./.venv/bin/python kernel-lab.py --engine mlx capture --workspace /tmp/mlx-kernel-workspace-2 --output /tmp/mlx-kernel-workspace-2/block-prelude-trace.gputrace --quick`
  - `./.venv/bin/python kernel-lab.py --engine mlx profile --preset m5-balanced --top-k 6`
  - `./.venv/bin/python kernel-lab.py --engine mlx orchestrate --profile /tmp/mlx-kernel-profile-ledger.json --workspace-root /tmp/mlx-kernel-orch-ledger --rank 1`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - `block_prelude` now records:
    - `verify_ok_count=1`
    - `capture_ok_count=1`
    - `promotion_status=ready-for-integration-test`
  - The same target moved to rank `1` for `m5-balanced`, with:
    - `base_priority_score=8.125`
    - `effective_priority_score=8.875`
  - The resulting orchestration plan upgraded from "open a new workspace" to:
    - `status=promotion-ready`
    - `workspace=/tmp/mlx-kernel-workspace-2`

### March 12, 2026 — `03f729b` — lab: Feed MLX trace artifacts back into orchestration — score `2`

**AI-identified within brief, human-approved (2)**

- Made the MLX orchestration layer consume trace metadata as an optional second input, so a target can move from "heuristically ranked" to "trace-backed investigation" without leaving the shared kernel-lab workflow.
  - Meaning: the trace layer is no longer just an isolated capture command. Once a workspace has a real `.gputrace`, the next orchestration plan can carry that evidence and explicitly tell the user to inspect the trace before editing and recapturing.
  - Motivation: after capture mode landed, the next gap was that trace artifacts were still detached from the normal lab loop. The fastest useful improvement was to keep profile-based target selection, but let real capture evidence shape the next recommended workflow.
  - Purpose: close the loop between heuristic ranking and trace-backed reality, so the kernel lab can escalate a target from "candidate" to "worth serious investigation" without inventing a separate manual process.
  - Added trace-metadata loading and summary helpers in `autoresearch_mlx/lab_trace.py`.
  - Extended MLX orchestration to accept `--trace-metadata`, validate target alignment, and emit a trace-backed workflow that includes inspecting the existing `.gputrace` and recapturing after edits.
  - Updated the README, kernel-lab note, architecture report, and generic agent prompt so the trace-backed orchestration loop is visible in the public workflow.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_lab/labs.py`
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_profile.py`
  - `autoresearch_mlx/lab_trace.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py autoresearch_mlx/lab_profile.py autoresearch_mlx/lab_trace.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx orchestrate --profile /tmp/mlx-kernel-profile.json --workspace-root /tmp/mlx-kernel-orch-trace --rank 2 --trace-metadata /tmp/mlx-kernel-workspace-2/block-prelude-trace.metadata.json`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - The trace-backed orchestration plan for `block_prelude` completed successfully with:
    - `status=trace-backed`
    - trace target match: `block_prelude`
    - embedded trace metric: `throughput_gb_s=4.328098329488359`
    - embedded bench `max_abs_error=0.0`

### March 12, 2026 — `8adbf2d` — lab: Add MLX capture mode and trace artifacts — score `3` — complexity `7`

**AI-identified within brief, human-shaped (3)**

- Split the MLX kernel-lab workflow into an explicit heuristic layer and trace layer, and added the first MLX capture path with real `.gputrace` artifacts.
  - Meaning: the kernel lab no longer treats `profile` and microbench results as if they were the whole profiling story. It now has a separate trace-backed path for Apple Silicon work where the artifact you inspect in Xcode is the truth source and the heuristic/profile layer is only the candidate-selection layer.
  - Motivation: after the first `profile / extract / orchestrate / verify` loop landed, the main blind spot was Apple traceability. The next useful step was to stop implying that model-aware heuristics are authoritative and add a first-class MLX capture flow that fits the actual Xcode/Instruments-shaped profiling ecosystem.
  - Purpose: make the MLX lab trustworthy enough for real kernel work by keeping fast candidate selection while also producing trace artifacts that can validate synchronization, queue pacing, hidden copies, and other backend realities that a synthetic microbench misses.
  - Added `LabTraceResult`, `supports_capture`, and `capture_workspace(...)` to the shared kernel-lab boundary in `autoresearch_lab/labs.py`.
  - Added `autoresearch_mlx/lab_trace.py` to own the MLX capture implementation, `.gputrace` path handling, and sidecar metadata emission.
  - Added MLX `capture` and internal `_capture-bench` commands in `autoresearch_mlx/lab.py`, and wired the MLX lab workspace to expose capture through the shared lab interface.
  - Updated the README, kernel-lab note, architecture report, and generic agent prompt so they now describe the heuristic layer as "choose the next target" and the trace layer as "validate what really happened."

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_lab/labs.py`
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_trace.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py autoresearch_mlx/lab_profile.py autoresearch_mlx/lab_trace.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx capture --workspace /tmp/mlx-kernel-workspace-2 --output /tmp/mlx-kernel-workspace-2/block-prelude-trace.gputrace --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - The first trace-backed capture completed successfully for the extracted `block_prelude` workspace:
    - trace artifact: `/tmp/mlx-kernel-workspace-2/block-prelude-trace.gputrace`
    - metadata sidecar: `/tmp/mlx-kernel-workspace-2/block-prelude-trace.metadata.json`
    - captured target: `block_prelude`
    - trace status: `ok`
    - trace wall time: `0.895s`
    - embedded bench result: `max_abs_error=0.0`, `median_latency_ms=6.861`, `median_throughput_gb_s=4.328`

### March 12, 2026 — `08832c3` — lab: Add MLX profile, extract, orchestrate, and verify workflow — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Added the first shared workflow layer above `list / init / bench`, with MLX implementations of `profile`, `extract`, `orchestrate`, and `verify`.
  - Meaning: the kernel lab is no longer just a catalog of starter workspaces. It now has the beginning of a repeatable outer loop that can discover likely targets, turn them into workspaces, and emit the next ready kernel-work command sequence.
  - Motivation: after the target catalog reached real training-path coverage, the next useful step was to stop broadening the catalog and start building the profile/extract/orchestrate layer that future MLX, Triton/CUDA, ROCm, and ANE labs can share.
  - Purpose: create the first backend-agnostic outer workflow for kernel work, while keeping MLX-specific heuristics and starter extraction logic in MLX-owned code.
  - Added shared profile/extract/orchestration result types and capability flags in `autoresearch_lab/labs.py`.
  - Added MLX profile heuristics in `autoresearch_mlx/lab_profile.py`, ranking likely targets from real preset/model structure instead of a static target list.
  - Added MLX `profile`, `extract`, `orchestrate`, and `verify` commands in `autoresearch_mlx/lab.py` and `autoresearch_mlx/lab_workspace.py`, including workspace metadata/context capture from saved profile artifacts.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_lab/labs.py`
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_profile.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py autoresearch_mlx/lab_profile.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx profile --preset m5-balanced --top-k 8`
  - `./.venv/bin/python kernel-lab.py --engine mlx profile --preset m5-balanced --top-k 6 --output /tmp/mlx-kernel-profile.json`
  - `./.venv/bin/python kernel-lab.py --engine mlx extract --profile /tmp/mlx-kernel-profile.json --workspace /tmp/mlx-kernel-workspace-2 --rank 2`
  - `./.venv/bin/python kernel-lab.py --engine mlx orchestrate --profile /tmp/mlx-kernel-profile.json --workspace-root /tmp/mlx-kernel-orch --rank 2`
  - `./.venv/bin/python kernel-lab.py --engine mlx verify --workspace /tmp/mlx-kernel-workspace-2 --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - `profile --preset m5-balanced --top-k 8` ranked the first MLX targets as:
    - `fused_mlp` (`priority_score=8.5`)
    - `block_prelude` (`priority_score=8.125`)
    - `attention_prelude` (`priority_score=7.325`)
  - `verify` on the extracted `block_prelude` workspace completed with `max_abs_error=0.0`, `median_latency_ms=6.044`, and `median_throughput_gb_s=5.553`.

### March 11, 2026 — `2c6bacc` — lab: Add composed loss and block-prelude starter targets — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Extended the MLX lab from support-path fragments into the next two composed targets: a full loss-side cross-entropy path and a bounded block-level prelude that combines residual blending, RMSNorm, and attention staging.
  - Meaning: the lab now has starter targets that represent complete composed subpaths, not just individual reshape/cast/gating pieces.
  - Motivation: after the support-path fragments were in place, the next useful step was to test whether the lab could represent broader real training-path clusters without jumping all the way to full attention or entire blocks.
  - Purpose: create the first composed targets that are large enough to motivate later profiling/orchestration work while still staying below the complexity threshold of full attention or full-block reimplementation.
  - Added `cross_entropy_full` for the final logits cast + softcap + cross-entropy + byte-aware masked reduction path.
  - Added `block_prelude` for residual blend + RMSNorm + attention staging up to, but not including, SDPA and output projection matmuls.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx list-targets`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target cross_entropy_full --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target block_prelude --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx bench --workspace <tmp> --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - Quick MLX lab benchmarks for the two new composed starter targets both completed with `max_abs_error=0.0`:
    - `cross_entropy_full`: `median_latency_ms=5.743`, `median_throughput_gb_s=13.485`
    - `block_prelude`: `median_latency_ms=2.509`, `median_throughput_gb_s=12.604`

### March 11, 2026 — `6aaac43` — lab: Add reshape and broader attention/logits starter targets — score `3` — complexity `7`

**AI-identified within brief, human-shaped (3)**

- Extended the MLX lab into the next composed support-path targets: value-embed lookup reshaping, the final logits cast+softcap path, the attention output reshape path, and a broader attention prelude target that stops just before SDPA.
  - Meaning: the lab now covers more of the reshape/cast/staging work that surrounds the large kernels, not just the elementwise and reduction pieces inside them.
  - Motivation: after gating, mask, and loss-side reduction, the next useful additions were the common reshaping and pre-attention/post-head operations that can matter in end-to-end latency without requiring a full custom attention kernel.
  - Purpose: keep broadening the kernel-lab catalog in model-faithful increments so the eventual cross-backend lab boundary can reason about more of the real training path than isolated toy ops.
  - Added `ve_lookup_reshape` for the value-embed table lookup followed by reshape into KV heads.
  - Added `proj_head_reshape` for the attention output transpose + reshape just before the output projection.
  - Added `loss_logits_cast_softcap` for the final cast-to-float32 plus tanh softcap path.
  - Added `attention_prelude` for the composed rotary + Q/K norm + transpose staging up to, but not including, scaled dot-product attention.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx list-targets`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target ve_lookup_reshape --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target proj_head_reshape --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target loss_logits_cast_softcap --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target attention_prelude --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx bench --workspace <tmp> --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - Quick MLX lab benchmarks for the four new support-path starter targets all completed with `max_abs_error=0.0`:
    - `ve_lookup_reshape`: `median_latency_ms=0.346`, `median_throughput_gb_s=31.959`
    - `proj_head_reshape`: `median_latency_ms=0.520`, `median_throughput_gb_s=9.149`
    - `loss_logits_cast_softcap`: `median_latency_ms=20.332`, `median_throughput_gb_s=29.686`
    - `attention_prelude`: `median_latency_ms=1.594`, `median_throughput_gb_s=12.563`

### March 11, 2026 — `cedaa52` — lab: Add value-gate, mask, and loss-side starter targets — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Extended the MLX kernel lab into the next support-path starter targets: the value-embed gate, local attention-mask construction, and the byte-aware loss-side reduction around flattened cross-entropy output.
  - Meaning: the lab now covers real support-path work from attention and loss evaluation instead of only forward-path math and backward norms.
  - Motivation: after forward-path and backward-norm coverage, the next useful additions were kernels that represent real model plumbing and loss-side work without jumping directly to reshape-heavy composed kernels or full attention.
  - Purpose: keep broadening the MLX lab along the actual training path so the later cross-backend lab boundary has starter patterns for attention gating, masking, and loss-side reduction too.
  - Added `value_embed_gate` for the gate-and-add path around learned value embeddings inside attention.
  - Added `attention_mask_local` for local-causal mask construction at a single window size.
  - Added `cross_entropy_prelude` for the byte-aware masked reduction around flattened loss output and target-byte counting.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx list-targets`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target value_embed_gate --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target attention_mask_local --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target cross_entropy_prelude --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx bench --workspace <tmp> --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - Quick MLX lab benchmarks for the three support-path starter targets all completed with `max_abs_error=0.0`:
    - `value_embed_gate`: `median_latency_ms=0.481`, `median_throughput_gb_s=15.150`
    - `attention_mask_local`: `median_latency_ms=0.536`, `median_throughput_gb_s=14.105`
    - `cross_entropy_prelude`: `median_latency_ms=0.496`, `median_throughput_gb_s=4.913`

### March 11, 2026 — `3ac1984` — lab: Add backward norm starter targets and tuple-aware benching — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Extended the MLX kernel lab into the first backward-capable starter targets, and generalized the bench harness so kernels can return multiple outputs with different shapes.
  - Meaning: the lab can now benchmark tuple-valued kernels cleanly and includes `rmsnorm_backward` and `layernorm_backward` as starter-ready targets rather than stopping at forward-only kernels.
  - Motivation: the next meaningful step after forward-path kernels is training-relevant backward work, and the existing harness was too single-output oriented to support realistic backward kernels cleanly.
  - Purpose: move the MLX lab closer to real training-path optimization while keeping the starter workflow simple and reusable for later CUDA/Triton, ROCm, and ANE lab backends.
  - Added `rmsnorm_backward` and `layernorm_backward` starter templates with analytical backward references rather than finite-difference approximations.
  - Generalized the fixed bench harness to flatten tuple/list outputs for correctness checks and `mx.eval`, so future multi-output kernels do not need one-off benchmark code.
  - Updated the public MLX target lists so the new backward path is visible from the top-level docs.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx list-targets`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target rmsnorm_backward --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target layernorm_backward --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx bench --workspace <tmp> --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - Quick MLX lab benchmarks for the two new backward starter targets both completed with `max_abs_error=0.0`:
    - `rmsnorm_backward`: `median_latency_ms=1.674`, `median_throughput_gb_s=6.167`
    - `layernorm_backward`: `median_latency_ms=2.815`, `median_throughput_gb_s=4.806`

### March 11, 2026 — `a4524aa` — lab: Add next MLX training-path starter targets — score `3` — complexity `8`

**AI-identified within brief, human-shaped (3)**

- Extended the MLX kernel lab from generic math kernels into the next set of real training-path kernels: residual blend, residual+RMSNorm, Q/K RMSNorm, RoPE+Q/K fused, logits softcap, and the pointwise activation stage.
  - Meaning: the MLX lab can now target much more of the actual forward path used by the model, not just isolated norms and generic primitives.
  - Motivation: after proving the starter pattern on norms, reductions, softmax, and fused MLP, the next useful step was to cover repeated block-local operations that are closer to real training bottlenecks and would still make sense later in Triton/CUDA, ROCm, or ANE labs.
  - Purpose: make the kernel lab more useful as a real optimization workspace for the training path, while still staying well short of full attention or optimizer-kernel complexity.
  - Added `residual_blend` for the per-layer `resid_lambda * x + x0_lambda * x0` path.
  - Added `residual_rmsnorm` to capture the natural fused follow-up to residual blending.
  - Added `qk_rmsnorm` and `rope_qk_fused` so the lab can now cover the post-RoPE Q/K normalization path directly.
  - Added `logits_softcap` for the final tanh-based logits clamp and `activation_pointwise` for the squared-ReLU activation stage inside the MLP.
  - Updated the public kernel-lab docs/catalogs so the visible MLX target set stays aligned with the actual workspace support.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --engine mlx list-targets`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target residual_blend --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target residual_rmsnorm --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target qk_rmsnorm --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target rope_qk_fused --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target logits_softcap --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target activation_pointwise --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx bench --workspace <tmp> --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - Quick MLX lab benchmarks for the six new training-path starter targets all completed with `max_abs_error=0.0`:
    - `residual_blend`: `median_latency_ms=0.545`, `median_throughput_gb_s=14.587`
    - `residual_rmsnorm`: `median_latency_ms=0.467`, `median_throughput_gb_s=9.826`
    - `qk_rmsnorm`: `median_latency_ms=1.492`, `median_throughput_gb_s=6.726`
    - `rope_qk_fused`: `median_latency_ms=2.595`, `median_throughput_gb_s=6.013`
    - `logits_softcap`: `median_latency_ms=21.573`, `median_throughput_gb_s=9.245`
    - `activation_pointwise`: `median_latency_ms=1.840`, `median_throughput_gb_s=12.644`

### March 11, 2026 — `b8f91ba` — lab: Expand MLX starter targets and rename top-level lab entrypoint — score `4` — complexity `10`

**Human-directed, AI-shaped (4)**

- Requested working through the practical MLX lab targets in order after the first RMSNorm foundation, with an eye toward the same top-level lab eventually hosting Triton/CUDA and other backend labs too, and then clarified that the public front door should be named like a real subsystem rather than `lab.py`.
  - Meaning: expanded the MLX lab from a single RMSNorm starter into a small starter-ready catalog with `layernorm`, `rotary_embedding`, `reduce`, `softmax`, and `fused_mlp`, all using the same mutable-workspace pattern and fixed benchmark harness, and renamed the public top-level entrypoint from `lab.py` to `kernel-lab.py`.
  - Motivation: the initial lab boundary was real, but still too narrow to prove the pattern would scale beyond one norm kernel, and once the lab became a first-class top-level system, `lab.py` was too generic and easy to confuse with unrelated experimentation or helper scripts.
  - Purpose: turn the MLX lab into a real backend workspace system that is useful now, serves as a believable template for future Triton/CUDA, ROCm, and ANE labs, and has a public front door whose name matches the rest of the repo's named subsystems.
  - Refactored the MLX lab harness around target-specific specs so new targets can bring their own templates, input generation, reference implementations, cases, and metrics without growing one giant conditional bench function.
  - Promoted `layernorm`, `rotary_embedding`, `reduce`, `softmax`, and `fused_mlp` to `starter-ready` targets with generated mutable workspaces and fixed quick/full benchmark cases.
  - Added `softmax` as an explicit starter target, since it is both common and numerically sensitive enough to justify a dedicated lab rather than hiding inside a broader MLP or attention target.
  - Hardened the bench harness so non-finite outputs now fail loudly instead of accidentally slipping through the error check, and scaled the fused-MLP random inputs/weights into a realistic range so the starter implementation can be benchmarked meaningfully in float16.
  - Renamed the public entrypoint to `kernel-lab.py`, updated the dispatcher/help text, and rewired the top-level docs so the kernel-lab surface now matches the rest of the repo's named subsystems.
  - Kept the backend implementation filenames under `autoresearch_*` unchanged so only the public front door changed.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `docs/platform-calibration.md`
  - `kernel-lab.py`
  - `autoresearch_lab/entrypoints.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile kernel-lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python kernel-lab.py --list-engines`
  - `./.venv/bin/python kernel-lab.py --engine mlx list-targets`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target layernorm --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target rotary_embedding --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target reduce --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target softmax --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx init --target fused_mlp --workspace <tmp>`
  - `./.venv/bin/python kernel-lab.py --engine mlx bench --workspace <tmp> --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - Quick MLX lab benchmarks for the expanded starter set all completed with `max_abs_error=0.0`:
    - `layernorm`: `median_latency_ms=0.900`, `median_throughput_gb_s=7.882`
    - `rotary_embedding`: `median_latency_ms=0.848`, `median_throughput_gb_s=5.852`
    - `reduce`: `median_latency_ms=0.496`, `median_throughput_gb_s=10.955`
    - `softmax`: `median_latency_ms=0.921`, `median_throughput_gb_s=7.015`
    - `fused_mlp`: `median_latency_ms=1.588`, `median_throughput_tflops=2.525`

### March 11, 2026 — `5592d01` — lab: Add shared kernel-lab boundary and first MLX workspace — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested borrowing the `autokernel` pattern for MLX work, but with an eye toward a top-level lab that could later host Triton/CUDA and other backend-specific kernel workflows under one shared outer structure.
  - Meaning: added a top-level `lab.py` entrypoint, a shared `autoresearch_lab/` boundary, and the first MLX implementation under `autoresearch_mlx/` with a mutable-workspace pattern plus a fixed benchmark harness.
  - Motivation: backend-specific kernel work was going to become important, but it did not fit cleanly into either the training-engine boundary or the platform-calibration path. Without a shared lab boundary, future MLX, Triton/CUDA, ROCm, or ANE kernel work would drift into separate ad hoc workflows.
  - Purpose: create a first-class experimental subsystem for kernel optimization that matches the repo's broader pattern: one generic top-level front door, backend-specific implementations underneath, and room to expand to more engines without creating new top-level orchestration trees each time.
  - Added `autoresearch_lab/` as the shared lab boundary and dispatch layer, parallel to the way `autoresearch_platform/` handles training engines.
  - Added the first MLX lab implementation with a starter-ready `rmsnorm` target, a generated mutable workspace, and a fixed correctness/performance benchmark harness.
  - Integrated the lab into the top-level docs and architecture story so `lab.py` now sits alongside `prepare.py`, `train.py`, and `calibrate.py` as a real public entrypoint rather than an orphaned experiment.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/kernel-lab.md`
  - `docs/mlx-port-architecture.md`
  - `docs/platform-calibration.md`
  - `lab.py`
  - `autoresearch_lab/__init__.py`
  - `autoresearch_lab/entrypoints.py`
  - `autoresearch_lab/labs.py`
  - `autoresearch_mlx/lab.py`
  - `autoresearch_mlx/lab_workspace.py`
- Validation:
  - `python3 -m py_compile lab.py autoresearch_lab/*.py autoresearch_mlx/lab.py autoresearch_mlx/lab_workspace.py`
  - `./.venv/bin/python lab.py --list-engines`
  - `./.venv/bin/python lab.py --engine mlx list-targets`
  - `./.venv/bin/python lab.py --engine mlx init --target rmsnorm --workspace /tmp/mlx-rmsnorm-lab-48983`
  - `./.venv/bin/python lab.py --engine mlx bench --workspace /tmp/mlx-rmsnorm-lab-48983 --quick`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - The first MLX lab workspace completed a quick RMSNorm benchmark with `max_abs_error=0.0`, `median_latency_ms=0.500`, and `median_throughput_gb_s=9.516` across the two quick benchmark cases.

### March 11, 2026 — `5c9ee53` — docs: Rename project to autoresearch-everywhere — score `4` — complexity `6`

**Human-directed, AI-shaped (4)**

- Requested renaming the GitHub repo and local project directory to `autoresearch-everywhere`, and updating the docs so the project actually presents itself under that new name.
  - Meaning: renamed the remote repository to `Entrpi/autoresearch-everywhere`, moved the local workspace directory to `/Users/ent/Codex/autoresearch-everywhere`, rewrote stale absolute-path references in the changelog, and updated the primary project docs to describe the repo as `autoresearch-everywhere` rather than a generic unnamed fork.
  - Motivation: once the repo name changed, leaving the local workspace, artifact links, and top-level docs under the old identity would make the rename feel partial and sloppy.
  - Purpose: make the project name, repo location, and user-facing documentation line up cleanly so future references, file links, and onboarding copy all point at the same identity.
  - Updated the main user-facing identity in `README.md`, `program.md`, and `docs/mlx-port-architecture.md`.
  - Rewrote the old absolute-path artifact and command references in `CHANGELOG.md` to the new local repo path.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - No runtime measurements; this was a repository-identity and documentation consistency change.

### March 11, 2026 — `7a4da10` — calibration: Restore trusted MLX eval calibration after engine-boundary refactor — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Inside the broader “close MLX regressions before finishing the refactor” workstream, surfaced the seeded-signature mismatch as the concrete parity bug and restored the trusted MLX eval rows to the current signatures.
  - Meaning: re-stamped the checked-in MLX eval calibration rows to the current eval/runtime signatures so known-hardware MLX runs once again resolve to calibrated `cheap/reference/full` policy instead of falling straight into `signature-mismatch` fallback after the file/layout refactor.
  - Motivation: the refactor had preserved the mechanics of the MLX workflow but broken trust in the seeded calibration table, which meant the new top-level path no longer behaved like the previous MLX-first system on its reference machine.
  - Purpose: restore a true parity baseline so the consolidated top-level entrypoints and one-button bring-up path retain the trusted calibrated behavior that made the earlier MLX-first system practical.
  - Verified that top-level MLX training now auto-selects calibrated eval rungs again on the reference M5 instead of falling back to default canonical eval.
  - Re-exercised the bounded one-button MLX bring-up path to confirm the restored calibration rows propagate cleanly through `prepare.py`, `train.py`, and `calibrate.py`.

**Grounding**

- Files:
  - `autoresearch_mlx/eval_policy.py`
  - `CHANGELOG.md`
  - Validation:
  - `./.venv/bin/python train.py --engine mlx --preset m5-fast --time-budget 0.2 --no-checkpoint`
  - `./.venv/bin/python train.py --engine mlx --preset m5-balanced --time-budget 20 --no-checkpoint`
  - `./.venv/bin/python prepare.py --num-shards 1`
  - `./.venv/bin/python train.py --smoke`
  - `./.venv/bin/python calibrate.py --engine mlx --mode fast --coarse-time-budget 0.5 --ranking-time-budget 0.5 --local-search-time-budget 0.5 --eval-train-seconds 0.5 --eval-rungs cheap,reference --output-dir /tmp/autoresearch_mlx_postfix_fast --force`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - `train.py --engine mlx --preset m5-fast --time-budget 0.2 --no-checkpoint` now reports `eval_calibration_status=calibrated` and `canonical_rung=cheap` for `m5-fast_apple-m5-32gb-10gpu`, where the same path had previously reported `signature-mismatch`.
  - `train.py --engine mlx --preset m5-balanced --time-budget 20 --no-checkpoint` now reports `eval_calibration_status=calibrated`, `eval_calibration_effective_confidence=telemetry-cross-session-stable`, and `canonical_rung=cheap`, while sustaining `steady_state_tok_per_sec=45055.8`.
  - The bounded full-workflow bring-up at `/tmp/autoresearch_mlx_postfix_fast/report.json` completed end to end after the fix and emitted a promotion bundle under `/tmp/autoresearch_mlx_postfix_fast/promotion`.

### March 11, 2026 — `0fe4594` — platform: Add shared MLX/CUDA training-engine boundary — score `4` — complexity `15`

**Human-directed, AI-shaped (4)**

- Requested that the repo stop treating platform calibration as MLX-only and instead grow a real training-engine boundary, with CUDA first, ROCm later, and ANE after that.
  - Meaning: added a shared engine contract under `autoresearch_platform/`, a first MLX engine adapter, a first CUDA engine adapter, and a safe CUDA runtime/config layer that can fingerprint NVIDIA hardware and surface architecture-specific capability metadata such as Hopper vs Blackwell flash-attention generation needs.
  - Motivation: the one-button bring-up story was no longer enough on its own. Without a broader engine boundary, every new backend would still have required its own training-stack glue, and even CUDA was still effectively "the old file" instead of a participant in the calibrated platform story.
  - Purpose: make the stack backend-extensible at the training-engine boundary so MLX and CUDA can share the same core hooks for train probes, local search, checkpoint minting, eval calibration, runtime capability reporting, and later promotion flow, while ROCm and ANE can follow the same path later.
  - Refactored `tools/calibrate_platform.py` to select an engine with `--engine`, use engine-owned preset catalogs and hardware fingerprints, and capability-gate local search, checkpoint minting, eval-rung calibration, and promotion output instead of assuming the MLX path everywhere.
  - Promoted `calibrate.py` to the top level as the public one-button bring-up entrypoint, while keeping `tools/calibrate_platform.py` as the implementation module underneath.
  - Added `autoresearch_cuda/config.py` so the CUDA defaults are importable without accidentally executing the CUDA trainer or pulling MLX-specific calibration code into the dependency chain.
  - Added `autoresearch_cuda/runtime.py` so CUDA architecture metadata is explicit: the runtime now treats A100/SM80, Ada RTX 40xx, Ada L40S-class, Hopper/SM90, RTX 50xx-class consumer Blackwell, B200-class Blackwell, and GB10/DGX Spark as distinct reference families, surfaces the preferred flash-attention generation for each current family, and carries an anticipated Vera Rubin family as a future slot without pretending its final capability or FA policy is already known.
  - Gave `autoresearch_cuda/train.py` a narrow CLI override surface for preset, time budget, sequence length, depth, window pattern, and batch shape so the CUDA engine can run comparable train probes through the same orchestration layer.
  - Made the CUDA trainer report architecture/runtime metadata and steady-state throughput in its final summary so engine-level consumers do not have to scrape the live progress line.
  - Consolidated the public top-level surface around `prepare.py`, `train.py`, `calibrate.py`, and `program.md`, with thin engine-dispatch wrappers at the top level and the concrete CUDA implementation moved under `autoresearch_cuda/`.
  - Made the consolidated surface actually usable as a front door by ensuring the generic wrapper help paths work even when secondary-engine runtime dependencies are absent, and by rewriting the primary README / generic agent instructions around the top-level entrypoints instead of MLX-specific script names.
  - Finished the root cleanup by moving the remaining MLX-specific top-level files into engine or support directories: `train_mlx.py` -> `autoresearch_mlx/train.py`, `prepare_mlx.py` -> `autoresearch_mlx/prepare.py`, `program_mlx.md` -> `docs/program-mlx.md`, `analysis.ipynb` -> `notebooks/analysis.ipynb`, `progress.png` -> `docs/assets/progress.png`, and `results.tsv` -> `results/results.tsv`.
  - Generalized the platform and architecture docs around the new engine boundary so the repo is no longer described as an MLX-only calibration stack.
  - Kept the boundary honest: MLX is still the only engine with full eval-calibration and promotion support, while CUDA is documented as the first narrower secondary engine on the same contract rather than being presented as feature-complete.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/mlx-port-architecture.md`
  - `docs/platform-calibration.md`
  - `program.md`
  - `docs/program-mlx.md`
  - `autoresearch_mlx/prepare.py`
  - `autoresearch_mlx/train.py`
  - `notebooks/analysis.ipynb`
  - `docs/assets/progress.png`
  - `results/results.tsv`
  - `calibrate.py`
  - `prepare.py`
  - `tools/calibrate_platform.py`
  - `train.py`
  - `autoresearch_platform/entrypoints.py`
  - `autoresearch_platform/__init__.py`
  - `autoresearch_platform/engines.py`
  - `autoresearch_platform/mlx_engine.py`
  - `autoresearch_platform/cuda_engine.py`
  - `autoresearch_cuda/__init__.py`
  - `autoresearch_cuda/config.py`
  - `autoresearch_cuda/prepare.py`
  - `autoresearch_cuda/runtime.py`
  - `autoresearch_cuda/train.py`
  - Validation:
  - `python3 -m py_compile prepare.py train.py calibrate.py autoresearch_platform/entrypoints.py tools/calibrate_platform.py autoresearch_platform/engines.py autoresearch_platform/mlx_engine.py autoresearch_platform/cuda_engine.py autoresearch_cuda/config.py autoresearch_cuda/prepare.py autoresearch_cuda/runtime.py autoresearch_cuda/train.py`
  - `./.venv/bin/python prepare.py --list-engines`
  - `./.venv/bin/python train.py --list-engines`
  - `./.venv/bin/python prepare.py --engine mlx --help`
  - `./.venv/bin/python train.py --engine mlx --help`
  - `./.venv/bin/python prepare.py --engine cuda --help`
  - `./.venv/bin/python train.py --engine cuda --help`
  - `./.venv/bin/python -c "from autoresearch_platform.engines import available_engines, get_engine; print(available_engines()); print(get_engine('mlx').default_platform_presets()); print(get_engine('cuda').reference_preset)"`
  - `./.venv/bin/python -c "from autoresearch_cuda.runtime import detect_cuda_runtime_profile; print(detect_cuda_runtime_profile((9,0))); print(detect_cuda_runtime_profile((10,0)))"`
  - `./.venv/bin/python calibrate.py --help`
  - `./.venv/bin/python calibrate.py --engine mlx --mode fast --presets m5-fast,m5-balanced --coarse-time-budget 0.2 --ranking-time-budget 0.2 --local-search-time-budget 0.2 --eval-train-seconds 0.2 --eval-rungs cheap --output-dir /tmp/autoresearch_engine_boundary_smoke --force`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
- Measurements:
  - The MLX smoke bring-up still completed end to end through the new engine boundary and emitted a full report under `/tmp/autoresearch_engine_boundary_smoke/report.md`.
  - The engine registry now resolves both `mlx` and `cuda`, with MLX keeping the shipped preset families and CUDA exposing `upstream` as its first reference preset.
  - The CUDA runtime classifier now reports:
    - A100/SM80 -> preferred flash-attention generation `2`
    - Ada `(8.9)` -> preferred flash-attention generation `2`
    - Hopper `(9.0)` -> preferred flash-attention generation `3`
    - Blackwell `(10.0)` -> preferred flash-attention generation `4`, with a separate GB10/DGX Spark carve-out when the device name identifies that system class
  - The current boundary is intentionally asymmetric:
    - `mlx` supports local search, checkpoint minting, eval-rung calibration, and promotion output
    - `cuda` currently supports hardware fingerprinting and comparable train probes, but not checkpoint-backed eval calibration yet

### March 11, 2026 — `d641b72` — calibration: Add one-button platform bring-up tool — score `4` — complexity `20`

**Human-directed, AI-shaped (4)**

- Requested that the next stage become a one-button bring-up flow for new hardware, not just a growing set of calibration subcommands.
  - Meaning: added `tools/calibrate_platform.py` as a real orchestrator that fingerprints the machine, runs a coarse preset envelope, ranks feasible preset families with short comparable training runs, performs a local operating-point search inside the winner, mints a checkpoint there, calibrates eval rungs on that checkpoint, and emits a Markdown/JSON bring-up bundle with a candidate new default for the autoresearch stage on that hardware.
  - Motivation: the existing calibration stack had the right primitives but still assumed a human who already knew which preset family to target and which subcommands to sequence. The missing product layer was the actual bring-up experience for a new user on unfamiliar hardware.
  - Purpose: let a new user clone the repo, run one long calibration command, and get a grounded starting zone, a candidate new default, lower and upper bounds, and an explicit relationship to the M5 and upstream-style references.
  - Reused the existing calibration machinery instead of duplicating it: the platform tool builds around `autoresearch_mlx/train.py` probes plus the existing `eval-rungs` logic from `tools/calibrate_eval_policy.py`.
  - Added stage-aware `fast` and `full` bring-up modes so the same tool can serve as either a quick first-default finder or a longer recommendation pass.
  - Made the output directory resumable: each major phase now records an input-keyed JSON artifact and later reruns reuse those artifacts unless `--force` is set.
  - Replaced the earlier hand-tuned weighted family chooser with a frontier-based selector plus pressure-aware memory shaping: rank candidates on measured quality, throughput, and eval overhead, keep the primary Pareto front, then choose the point closest to the ideal measured frontier while treating memory as a small tie-break cost below `50%` of unified memory and a progressively real penalty as pressure rises.
  - Expanded the local operating-point search defaults by mode so `seq_len` and `window_pattern` can participate automatically in fuller bring-up runs without turning the tool into an architecture search.
  - Defined zone outputs concretely in the report using measured ranking probes rather than preset ordering: `lower`, `recommended`, `upper`, and `reference`, with `upstream` always treated as the reference zone when it runs successfully.
  - Made the candidate-default logic explicit rather than implicit: the report now emits a machine-readable default block with preset family, tuned operating point, selection confidence, and the measured selection components that selected it.
  - Added a promotion bundle to the output directory so the bring-up path no longer stops at a report: it now emits a platform-default artifact, a Python fragment for that default, and a calibration artifact that is explicitly marked promotable or incomplete.
  - Added report-time comparison to the checked-in M5 reference on both axes we currently have: train-side throughput/memory and eval-side rung economics for the chosen preset family.
  - Added a report-time comparison to the local upstream-style preset behavior so the user can see whether upstream is merely a reference, an upper bound, or already practical on the new machine.
  - Updated the platform-calibration docs from future-plan language to current implementation language, including the staged modes, phase reuse, measured-zone framing, and remaining limitations.
  - Reshaped the top-level README around the intended new-user path so unfamiliar hardware now goes through one-button bring-up first, while the known M5 reference path is presented as the shortcut.
  - Rewrote the MLX architecture report around the newer subsystem boundaries: platform bring-up orchestration, runtime eval policy and telemetry, training engine, data/evaluation substrate, and optional local sweep tooling.
  - Added explicit code-shape signatures to the calibration story so the same system now covers both new-hardware bring-up and post-change revalidation: eval rows and bring-up outputs are stamped with eval-semantics and runtime-shape signatures, and the runtime now falls back visibly when those no longer match.
  - Reframed the recalibration trigger so agent judgment is primary and signatures are only the conservative backstop: the operator docs now tell the agent to rerun platform calibration proactively after findings that look likely to generalize across preset shapes or hardware classes, instead of waiting for static signature drift to be the whole policy.
  - Removed `upstream` from the default one-button sweep set on MLX hardware. It remains available as an explicit reference via `--presets ...,upstream`, but the default bring-up path now focuses on the practical shipped MLX preset families so new-user calibration does not spend most of its wall time on a shape that is rarely the local default.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/mlx-port-architecture.md`
  - `docs/preset-calibration.md`
  - `docs/platform-calibration.md`
  - `autoresearch_mlx/calibration_signature.py`
  - `autoresearch_mlx/eval_policy.py`
  - `tools/calibrate_platform.py`
  - `tools/calibrate_eval_policy.py`
  - `autoresearch_mlx/train.py`
  - `docs/program-mlx.md`
- Validation:
  - `python3 -m py_compile tools/calibrate_platform.py tools/calibrate_eval_policy.py autoresearch_mlx/eval_policy.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/render_autonomy_badge.py`
  - `./.venv/bin/python tools/calibrate_platform.py --mode fast --presets m5-fast,m5-balanced,upstream --coarse-time-budget 0.2 --ranking-time-budget 0.2 --local-search-time-budget 0.2 --eval-train-seconds 0.2 --eval-rungs cheap,reference --output-dir /tmp/autoresearch_calibrate_platform_smoke2 --force`
  - `/usr/bin/time -p ./.venv/bin/python tools/calibrate_platform.py --mode fast --presets m5-fast,m5-balanced,upstream --coarse-time-budget 0.2 --ranking-time-budget 0.2 --local-search-time-budget 0.2 --eval-train-seconds 0.2 --eval-rungs cheap,reference --output-dir /tmp/autoresearch_calibrate_platform_smoke2`
- Measurements:
  - The reduced-budget end-to-end smoke run exercised the full phase chain successfully:
    - hardware fingerprint
    - coarse preset envelope
    - candidate ranking
    - local operating-point search
    - candidate checkpoint minting
    - final eval-rung calibration
    - Markdown + JSON report generation
  - The rerun without `--force` completed in `1.08s`, reusing the saved phase artifacts instead of replaying the whole bring-up sequence.
  - On that smoke run, the tool produced a coherent candidate default block:
    - candidate preset family: `m5-fast`
    - tuned operating point: `seq_len=256`, `window_pattern=L`, `device_batch_size=4`, `total_batch_size=2048`
    - family relation to M5 default: `smaller-than-m5-default`
    - selection confidence: `telemetry-repeated-single-hardware`
  - The same run emitted a promotion bundle under `/tmp/autoresearch_calibrate_platform_smoke3/promotion`:
    - platform default artifacts were immediately promotion-ready
    - eval calibration was explicitly marked `promotable=false` because the fast-mode run did not include the `full` rung
  - The same smoke report also produced the intended zone structure:
    - `m5-fast -> recommended`
    - `m5-balanced -> upper`
    - `upstream -> reference`
  - The candidate ranking now exposes the selection story in the report, including `Pareto` membership, `frontier_distance`, `selection_distance`, and memory-fraction pressure bands.
  - This was a bounded pipeline validation, not a final platform calibration. The smoke invocation used only `cheap,reference` eval rungs and `0.2s` training budgets, so the resulting default block is useful as a correctness check on the orchestration path, not as a production recommendation.

### March 11, 2026 — `871cc3f` — calibration: Harden runtime eval selection against semantic drift — score `4` — complexity `15`

**Human-directed, AI-shaped (4)**

- Requested that the calibration automation broaden from an eval-policy seed into a practical operating-point search for new preset / hardware combinations.
  - Meaning: `tools/calibrate_eval_policy.py` now exposes `train-grid`, a constrained training sweep that can cross `device_batch_size`, `total_batch_size`, `seq_len`, and `window_pattern`, while defaulting any omitted axes to the preset values. `autoresearch_mlx/train.py` now consumes the checked-in eval tradeoff table by default for shipped preset shapes when canonical eval settings are not manually overridden, but only on exact hardware-key matches.
  - Motivation: the foundation was still underscoped. Real preset calibration needs to identify local training operating points, not just cheap / reference / full eval rungs, and the runtime selector needs to make calibration gaps explicit instead of silently applying the M5 row everywhere.
  - Purpose: let new preset and hardware bring-up follow the same mechanical loop we have been doing manually: small training-grid search first, then eval batch and rung calibration on the chosen operating point, with the trainer automatically benefiting from measured rung tables only when the shape and hardware are actually covered.
  - Promoted the training sweep from a batch-only interface to a true operating-point grid, while keeping `train-batch` as an alias so the initial foundation commands still work.
  - Added `tokens_per_fwdbwd` and `grad_accum_steps` to each row so the accumulation shape is explicit in calibration artifacts rather than inferred from the arguments.
  - Rewrote the calibration docs around the intended workflow: `train-grid -> eval-batch -> eval-rungs`, with an explicit warning to keep the grid small and interpretable.
  - Made the runtime selector conservative: it only auto-selects cheap / reference / full for shipped preset shapes with no explicit canonical override, falls back to the default canonical settings for mutated shapes until they are calibrated, and also falls back when no exact hardware-key row exists.
  - Added explicit calibration metadata to runtime config and summaries: hardware key, calibration status, calibration key, confidence, repeat count, measurement budget, measurement date, and policy version.
  - Added a stale-row guard: if a checked-in calibration row carries the wrong policy version, the runtime now falls back visibly instead of silently trusting it.
  - Added passive eval telemetry capture for ordinary eligible runs, stored outside git, so repeat evidence can accumulate without hand-running the calibration tool every time.
  - Aggregated that telemetry into runtime freshness and richer effective-confidence signals, including telemetry count, commit/day spread, observed rungs, stable rungs, and last-seen date.
  - Made confidence policy-driving instead of descriptive: low-confidence seed rows can auto-pick `cheap` or `reference`, but they no longer auto-escalate to `full` without broader stable cross-rung evidence.
  - Added an age-based guard on top of schema versioning: stale-age rows now fall back visibly to the default canonical settings instead of being trusted indefinitely.
  - Added `telemetry-summary` to the calibration CLI so the passive evidence base can be inspected directly instead of inferred from trainer logs.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/preset-calibration.md`
  - `docs/program-mlx.md`
  - `autoresearch_mlx/eval_policy.py`
  - `autoresearch_mlx/eval_telemetry.py`
  - `autoresearch_mlx/constants.py`
  - `autoresearch_mlx/train.py`
  - `tools/calibrate_eval_policy.py`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/eval_policy.py autoresearch_mlx/eval_telemetry.py tools/calibrate_eval_policy.py`
  - `./.venv/bin/python tools/calibrate_eval_policy.py telemetry-summary --preset m5-balanced`
  - `./.venv/bin/python tools/calibrate_eval_policy.py train-grid --preset m5-fast --time-budget 0.2 --device-batches 1,2 --total-batches 512,1024 --json-out /tmp/calibrate_train_grid_batch_test.json`
  - `./.venv/bin/python tools/calibrate_eval_policy.py train-grid --preset m5-fast --time-budget 0.2 --device-batches 2 --total-batches 1024 --seq-lens 256,512 --json-out /tmp/calibrate_train_grid_seq_test.json`
  - `./.venv/bin/python tools/calibrate_eval_policy.py train-grid --preset m5-balanced --time-budget 1.5 --device-batches 4 --total-batches 2048 --window-patterns L,SSSL --json-out /tmp/calibrate_train_grid_window_test.json`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-fast --time-budget 0.2 --no-checkpoint`
  - `./.venv/bin/python - <<'PY' ... resolve_run_config(...) / choose_auto_eval_decision(...) ... PY` to verify default selection, long-run confidence capping, boosted-confidence promotion to `full`, and stale-age fallback
- Measurements:
  - The new `device_batch_size x total_batch_size` grid already surfaces a real operating-point choice on `m5-fast`:

    | device batch | total batch | grad accum | steady tok/s |   peak MB |
    | -----------: | ----------: | ---------: | -----------: | --------: |
    |        `1` |     `512` |      `2` |   `7296.8` | `107.1` |
    |        `2` |     `512` |      `1` |  `32493.6` | `147.0` |
    |        `1` |    `1024` |      `4` |  `28728.2` | `114.7` |
    |        `2` |    `1024` |      `2` |  `39978.7` | `156.5` |
  - The same command path now handles a constrained `seq_len` sweep without extra glue code:

    | seq len | device batch | total batch | grad accum | steady tok/s |   peak MB |
    | ------: | -----------: | ----------: | ---------: | -----------: | --------: |
    | `256` |        `2` |    `1024` |      `2` |  `23750.5` | `156.5` |
    | `512` |        `2` |    `1024` |      `1` |  `30684.2` | `253.2` |
  - And it can probe a small `window_pattern` candidate set on the same preset shell:

    | window   | steady tok/s |   peak MB |  steps |
    | -------- | -----------: | --------: | -----: |
    | `L`    |  `34663.7` | `950.2` | `23` |
    | `SSSL` |  `35698.2` | `950.2` | `26` |
  - The short `window_pattern` probe is only a capability check for the new axis, not a recommendation to change preset defaults. The important point is that batch and shape axes now live under one mechanical sweep instead of separate ad hoc scripts.
  - The richer aggregation policy now distinguishes repeated evidence from broad stable evidence:

    | case                                                 | rung          | status                 | effective confidence                   | freshness     | telemetry | stable rungs        | limited by     |
    | ---------------------------------------------------- | ------------- | ---------------------- | -------------------------------------- | ------------- | --------: | ------------------- | -------------- |
    | `m5-balanced`, `300s`                            | `reference` | `calibrated`         | `telemetry-repeated-single-hardware` | `fresh`     |    `42` | `reference`       |                |
    | `m5-balanced`, `8h`                              | `reference` | `calibrated-limited` | `telemetry-repeated-single-hardware` | `fresh`     |    `42` | `reference`       | `confidence` |
    | simulated broader stable evidence                    | `full`      | `calibrated`         | `telemetry-cross-session-stable`     | `fresh`     |     `8` | `cheap,reference` |                |
    | simulated stale row                                  | `default`   | `stale-age`          | `telemetry-cross-session-stable`     | `stale-age` |     `8` | `cheap,reference` | `stale-age`  |
    | `m5-balanced` with `--seq-len 1024`              | `default`   | `shape-fallback`     |                                        |               |           |                     |                |
    | `m5-balanced` on simulated `apple-m9-96gb-40gpu` | `default`   | `hardware-unmatched` |                                        |               |           |                     |                |
  - `telemetry-summary` exposes the passive evidence directly for a real preset/hardware row:

    - `m5-balanced` on `apple-m5-32gb-10gpu`: `eligible_count=42`, `commit_count=6`, `day_count=2`, `observed_rungs=reference`, `stable_rungs=reference`
    - rung stats: `median_eval_seconds=13.552`, `rel_mad_eval_seconds=0.0089`, `stable_timing=true`
  - A short real `m5-fast` run confirms that ordinary runs now append passive telemetry and feed it back into the next matching config:

    - `canonical_eval_rung=cheap`
    - run summary: `eval_calibration_status=calibrated`, `eval_calibration_effective_confidence=seed-single-checkpoint`, `eval_calibration_limited_by=None`
    - telemetry ledger after the run: `~/.cache/autoresearch/eval_policy_telemetry.jsonl` exists and increments
    - next config resolve for the same preset sees `eval_calibration_telemetry_count > 0`, `eval_calibration_commit_count > 0`, and `eval_calibration_last_seen_on=2026-03-10`
    - `eval_hardware_key=apple-m5-32gb-10gpu`
    - `eval_calibration_confidence=seed-single-checkpoint`
    - `canonical_eval_seq_len=2048`
    - `canonical_eval_tokens=262144`
    - `canonical_eval_batch_size=2`
    - `canonical_eval_slices=32`

### March 10, 2026 — `760fa75` — calibration: Add preset calibration tooling foundation — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested that the new calibration work cover batching and other model-shape / hardware-capability decisions, not just eval rung selection.
  - Meaning: the repo now has a first-pass calibration layer with a checked-in `eval_policy.py` for current M5 rung data and a `tools/calibrate_eval_policy.py` CLI that can sweep train-side device batches, eval batches, and cheap/reference/full eval rungs.
  - Motivation: an eval-only selector would underscope the real problem. We also need a mechanical way to identify train-side batch sweet spots and other preset / hardware operating points instead of rediscovering them manually.
  - Purpose: turn the recent manual `fast / balanced / large / xlarge` policy work into a repeatable calibration loop that can be rerun for new preset shapes and future hardware.
  - Kept the trainer behavior unchanged for now; this is an offline calibration layer, not a new default selector wired into `autoresearch_mlx/train.py`.
  - Seeded the policy module with the current M5 calibrations for `m5-fast`, `m5-balanced`, `m5-large`, and `m5-xlarge`.
  - Made the calibration CLI able to reuse existing checkpoints or mint a fresh short checkpoint for rung measurement, so it can be used incrementally rather than only as a full re-benchmark pass.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `docs/preset-calibration.md`
  - `autoresearch_mlx/eval_policy.py`
  - `tools/calibrate_eval_policy.py`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/eval_policy.py tools/calibrate_eval_policy.py`
  - `./.venv/bin/python tools/calibrate_eval_policy.py train-batch --preset m5-fast --time-budget 0.2 --device-batches 1,2 --json-out /tmp/calibrate_train_batch_test.json`
  - `./.venv/bin/python tools/calibrate_eval_policy.py eval-rungs --preset m5-fast --checkpoint /tmp/autoresearch_m5_fast_2min_eval_ladder --rungs cheap,reference --json-out /tmp/calibrate_eval_rungs_test.json`
  - `./.venv/bin/python tools/calibrate_eval_policy.py eval-batch --checkpoint /tmp/autoresearch_balanced_2min_compare --seq-len 2048 --eval-tokens 262144 --batches 1,2,4 --json-out /tmp/calibrate_eval_batch_default_test.json`
- Measurements:
  - The quick `train-batch` validation already reproduces the expected local batch preference on `m5-fast`:

    | device batch | status | steady tok/s |   peak MB |
    | -----------: | ------ | -----------: | --------: |
    |        `1` | `ok` |    `976.4` |  `86.6` |
    |        `2` | `ok` |  `26249.2` | `147.0` |
  - The `eval-rungs` mode reproduces the existing `m5-fast` rung measurements on the saved checkpoint without retraining:

    | rung          | eval sec |     `val_bpb` |
    | ------------- | -------: | --------------: |
    | `cheap`     | `1.32` | `2.002362005` |
    | `reference` | `6.54` | `2.014320405` |
  - The `eval-batch` mode surfaced an important calibration rule: batch sweeps should stay sequential by default. With the tool's intended default (`eval_slices=1`), the same saved `m5-balanced` checkpoint stays effectively batch-invariant at `seq=2048`:

    | batch | eval sec |     `val_bpb` |
    | ----: | -------: | --------------: |
    | `1` | `2.44` | `1.622281806` |
    | `2` | `2.47` | `1.622281811` |
    | `4` | `2.60` | `1.622281804` |
  - A sliced `eval-batch` probe during validation showed visible batch dependence on the same checkpoint, which is a useful failure mode to catch early:

    - when the batch sweep reused horizon slicing (`eval_slices=32`, `reference_eval_tokens=20971520`), the measured `val_bpb` drifted across batches instead of staying invariant
    - that is why the tool keeps `eval-batch` sequential by default, while `eval-rungs` is the place where reduced-budget sliced canonical measurements belong
  - The seeded policy module already produces the selector preview we would expect from the earlier manual tables:

    | preset          | `5m` recommendation     | `8h` recommendation      |
    | --------------- | ------------------------- | -------------------------- |
    | `m5-fast`     | `reference` (`2.58%`) | `full` (`0.381%`)      |
    | `m5-balanced` | `reference` (`5.78%`) | `full` (`0.638%`)      |
    | `m5-large`    | `cheap` (`1.96%`)     | `reference` (`0.121%`) |
    | `m5-xlarge`   | `cheap` (`2.18%`)     | `reference` (`0.132%`) |

### March 10, 2026 — `28fe7d9` — eval: Stratify canonical val sampling across the upstream horizon — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Requested that the cheap canonical eval stop behaving like a luck-of-the-prefix estimate and instead sample the same upstream-sized prefix horizon more representatively.
  - Meaning: canonical BPB now keeps the same token-loss math and `2048` context, but when prepacked val rows are available it samples evenly spaced contiguous slices across the first upstream-sized eval horizon instead of just reading one deterministic prefix.
  - Motivation: the earlier token sweep showed that the sequential-prefix estimator was strongly slice-biased, while a naïve whole-shard slice strategy overshot the upstream-style result. The missing piece was to stratify within the upstream horizon itself rather than across the whole val shard.
  - Purpose: get cheaper canonical numbers that track the full upstream-shaped eval more closely without paying the wall time of the full `40 * 524288` token contract.
  - Kept proxy eval sequential for now; only canonical eval uses the new horizon-aware slice strategy.
  - Derived the slice count from eval length instead of a fixed constant, using roughly `8` eval steps per slice and a cap of `32`.
  - Used the saved checkpoint sweeps to choose that rule rather than guessing a slice count by intuition.

**Grounding**

- Files:
  - `autoresearch_mlx/data.py`
  - `autoresearch_mlx/constants.py`
  - `autoresearch_mlx/train.py`
  - `README.md`
  - `docs/program-mlx.md`
- Validation:
  - Reused the saved `m5-balanced` 2-minute checkpoint at `/tmp/autoresearch_balanced_2min_compare` to isolate eval behavior from training variance.
  - Swept upstream-horizon slice counts at `seq=2048` for both the cheap canonical budget and the Trevin-sized budget.
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/constants.py autoresearch_mlx/data.py`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-balanced --time-budget 0.2 --eval-tokens 4096 --canonical-eval-tokens 4096 --no-checkpoint`
- Measurements:
  - On the same saved `m5-balanced` 2-minute checkpoint, the upstream-horizon slice sweep picked a clear rule:

    | eval contract                   |   steps | best slice count | best `val_bpb` |  eval sec |
    | ------------------------------- | ------: | ---------------: | ---------------: | --------: |
    | canonical `2048 / 262144 / 2` |  `64` |            `8` |  `1.625061687` |  `2.31` |
    | Trevin `2048 / 1572864 / 2`   | `384` |           `32` |  `1.627754259` | `14.80` |
  - Against the previously measured upstream practical reference (`2048 / 20971520 / 256 -> 1.627251` on this same checkpoint), the new slice strategy moved the cheap contracts closer without adding meaningful runtime:

    | eval contract                   | mode                    |     `val_bpb` | abs error vs upstream practical |  eval sec |
    | ------------------------------- | ----------------------- | --------------: | ------------------------------: | --------: |
    | canonical `2048 / 262144 / 2` | sequential prefix       | `1.622281811` |                    `0.004969` |  `2.38` |
    | canonical `2048 / 262144 / 2` | upstream-horizon slices | `1.625061687` |                    `0.002189` |  `2.37` |
    | Trevin `2048 / 1572864 / 2`   | sequential prefix       | `1.609068586` |                    `0.018182` | `14.16` |
    | Trevin `2048 / 1572864 / 2`   | upstream-horizon slices | `1.627754259` |                    `0.000503` | `14.13` |
  - Re-ran the full upstream-shaped baseline on the same checkpoint with the locally optimal long-context batch (`2`) under both sequential and sliced scheduling:

    | full upstream mode                  |     `val_bpb` |   eval sec |
    | ----------------------------------- | --------------: | ---------: |
    | sequential `2048 / 20971520 / 2`  | `1.627251004` | `183.79` |
    | sliced `2048 / 20971520 / 2 / 32` | `1.627251004` | `232.12` |
  - That confirmed the cheap sliced estimator is still chasing the same full upstream target rather than a different metric, and that full upstream eval should stay sequential because slicing only adds wall time at full budget.
  - Built the first preset-calibration table for an eventual rung selector using the same sliced cheap/reference contracts and the sequential full upstream baseline:

    | preset          | rung      |  eval tokens |   eval sec | abs error vs full | speedup vs full | `5m` overhead | `8h` overhead |
    | --------------- | --------- | -----------: | ---------: | ----------------: | --------------: | --------------: | --------------: |
    | `m5-fast`     | cheap     |   `262144` |   `1.34` |      `0.008803` |       `81.8x` |       `0.45%` |      `0.005%` |
    | `m5-fast`     | reference |  `1572864` |   `7.74` |      `0.003155` |       `14.2x` |       `2.58%` |      `0.027%` |
    | `m5-fast`     | full      | `20971520` | `109.74` |      `0.000000` |        `1.0x` |      `36.58%` |      `0.381%` |
    | `m5-balanced` | cheap     |   `262144` |   `2.96` |      `0.002189` |       `62.0x` |       `0.99%` |      `0.010%` |
    | `m5-balanced` | reference |  `1572864` |  `17.34` |      `0.000503` |       `10.6x` |       `5.78%` |      `0.060%` |
    | `m5-balanced` | full      | `20971520` | `183.79` |      `0.000000` |        `1.0x` |      `61.26%` |      `0.638%` |
    | `m5-large`    | cheap     |   `262144` |   `5.87` |      `0.004303` |       `74.0x` |       `1.96%` |      `0.020%` |
    | `m5-large`    | reference |  `1572864` |  `34.82` |      `0.003782` |       `12.5x` |      `11.61%` |      `0.121%` |
    | `m5-large`    | full      | `20971520` | `434.09` |      `0.000000` |        `1.0x` |     `144.70%` |      `1.507%` |
    | `m5-xlarge`   | cheap     |   `262144` |   `6.55` |      `0.008206` |       `72.6x` |       `2.18%` |      `0.023%` |
    | `m5-xlarge`   | reference |  `1572864` |  `38.03` |      `0.004042` |       `12.5x` |      `12.68%` |      `0.132%` |
    | `m5-xlarge`   | full      | `20971520` | `475.17` |      `0.000000` |        `1.0x` |     `158.39%` |      `1.650%` |
  - The early selector read is preset-sensitive rather than global:

    - `m5-fast` has a plausible middle rung; `reference` cuts error by about `2.8x` versus `cheap` while still costing only `2.58%` of a `5m` training run.
    - `m5-balanced` is the cleanest argument for the three-rung policy itself: `reference` is much closer to full than `cheap` (`0.000503` vs `0.002189` error) while still staying under `6%` overhead for a `5m` run, and full upstream only becomes cheap enough to treat as normal once the run is much longer.
    - `m5-large` does not have the same middle-rung economics on a `5m` run; `reference` is only slightly more accurate than `cheap`, but costs `11.61%` of the run budget instead of `1.96%`.
    - `m5-xlarge` behaves like the heavier version of that same story: `reference` does improve over `cheap` (`0.004042` vs `0.008206` error), but the `5m` tax is `12.68%`, so it still reads more like a long-run rung than a short-run default.
    - On long runs the tradeoff flips. For an `8h` training run, even `m5-large` `reference` is only `0.121%` overhead, while full upstream remains expensive enough (`1.507%`) that it still reads more like an audit rung than a default.
    - `m5-xlarge` reaches the same conclusion with slightly worse full-rung cost: `reference` is only `0.132%` overhead on an `8h` run, while full upstream is still `1.650%`.
  - This change is about estimator quality, not raw eval speed. Runtime stayed effectively flat while the cheap long-context estimates moved substantially closer to the upstream-shaped reference.

### March 10, 2026 — `a3c1aa2` — train: Scale canonical eval batch with sequence length — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested that canonical evaluation batch sizing follow the measured sequence-length sweep instead of staying fixed at `4`.
  - Meaning: canonical eval now defaults to a constant `4096` tokens per eval step, so batch size scales down as canonical sequence length scales up.
  - Motivation: the batch sweeps on the same saved `m5-balanced` checkpoint showed effectively identical BPB across batches, with a clean fastest pattern of `256 -> 16`, `512 -> 8`, `1024 -> 4`, and `2048 -> 2`.
  - Purpose: keep canonical eval cheap and consistent across sequence lengths without hand-tuning the batch every time the canonical eval shape changes.
  - Updated the trainer to auto-derive `canonical_eval_batch_size` from `canonical_eval_seq_len` when the user does not override it explicitly.
  - Raised the default canonical sequence length to `2048` and widened the live model config to `max(train seq, canonical seq)` so shorter-sequence presets can still use the upstream-shaped canonical eval.
  - Kept smoke runs and explicit `--canonical-eval-batch-size` overrides unchanged.

**Grounding**

- Files:
  - `autoresearch_mlx/constants.py`
  - `README.md`
  - `docs/program-mlx.md`
  - `autoresearch_mlx/train.py`
- Validation:
  - Reused the saved `m5-balanced` 2-minute checkpoint at `/tmp/autoresearch_balanced_2min_compare` to isolate eval behavior from training variance.
  - Swept canonical eval batches at `seq=256`, `seq=512`, `seq=1024`, and `seq=2048` with a hard `20s` timeout per candidate.
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/constants.py`
- Measurements:
  - On the same saved `m5-balanced` 2-minute checkpoint, the canonical-style BPB estimates tightened from shorter-sequence local estimates toward the upstream-shaped contract as sequence length increased:

    - `seq=256`: `1.660391`
    - `seq=512`: `1.634503`
    - `seq=1024`: `1.560247`
    - `seq=2048`, `262144` tokens: `1.622282`
    - `seq=2048`, Trevin contract (`1572864` tokens): `1.609069`
    - `seq=2048`, upstream practical contract (`20971520` tokens): `1.627251`
  - `seq=256`: fastest batch `16`
  - `seq=512`: fastest batch `8`
  - `seq=1024`: fastest batch `4`
  - `seq=2048`: fastest batch `2`
  - Under the Trevin-sized long-context contract (`seq=2048`, `eval_tokens=1572864`), the optimized local batch stayed metric-equivalent while materially reducing eval time:

    |   batch |  eval sec |     `val_bpb` |
    | ------: | --------: | --------------: |
    |   `2` | `14.02` | `1.609068586` |
    | `256` | `20.31` | `1.609068590` |
  - BPB was effectively invariant across the swept batch sizes for each sequence length, so the change is about eval efficiency, not metric drift.

### March 10, 2026 — `1778297` — changelog: Sync autonomy-golf resources from canonical repo — score `4` — complexity `8`

**Human-directed, AI-shaped (4)**

- Requested that `autoresearch` pull in the newer autonomy-golf resources from the canonical repo instead of carrying an older local copy.
  - Meaning: the repo should use the same manifesto, agent brief, checklist, parser, and badge flow as the canonical autonomy-golf bundle, while still keeping its local branch policy and MLX-specific guidance.
  - Motivation: the local copy had drifted behind the canonical `0..6` scale, maintenance checklist, README snapshot generation, and house-term badge language.
  - Purpose: keep `autoresearch` on the maintained autonomy-golf path so its scorekeeping and local docs stay compatible with the shared tooling.
  - Replaced the local autonomy-golf docs bundle with the current canonical manifesto and agent brief, and added the missing checklist.
  - Upgraded the local changelog parser and badge renderer to the canonical versions, including the generated README snapshot block and the house-term golf badge.
  - Aligned the local README, MLX agent prompt, and changelog headers to the canonical scoring and complexity model while preserving the branch rule that fully human-authored code changes belong in a fork.
  - Renamed the active changelog section to `Latest` and aligned the local parser, renderer, and guidance commands to `--include-latest`, so the wording reads more naturally while staying consistent with the synced tooling.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/autonomy-golf.md`
  - `docs/autonomy-golf-agent.md`
  - `docs/autonomy-golf-checklist.md`
  - `docs/program-mlx.md`
  - `tools/changelog_scores.py`
  - `tools/render_autonomy_badge.py`
  - `docs/autonomy-golf-badge.svg`
- Validation:
  - `python3 -m py_compile tools/changelog_scores.py tools/render_autonomy_badge.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/changelog_scores.py --group-by overall --format csv --include-latest`
  - `python3 tools/render_autonomy_badge.py`
- Measurements:
  - This is governance and scoring-tooling work, not a runtime optimization, so there are no performance measurements.

### March 9, 2026 — `518a595` — train: Add train and wall time budget modes — score `4` — complexity `8`

**Human-directed, AI-shaped (4)**

- Requested the budget split that had been discussed earlier: keep the original training-time objective for core research changes, but add a separate wall-clock mode for checkpointing and orchestration work.
  - Added explicit `train` vs `wall` budget accounting to `autoresearch_mlx/train.py` instead of overloading one stop condition to serve both goals.
  - Kept `train` as the default so the main autoresearch loop still optimizes on actual training time rather than incidental wall-clock blockage.
  - Added explicit reporting of which budget mode was active and how much budget-counted time elapsed in the invocation.
  - Included the budget mode in automatic checkpoint directory slugs so train-budget and wall-budget runs do not collide on the same auto checkpoint path.

**Grounding**

- Files:
  - `autoresearch_mlx/train.py`
  - `autoresearch_mlx/checkpoint_policy.py`
  - `README.md`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/checkpoint_policy.py`
  - `./.venv/bin/python - <<'PY' ...` to verify that automatic checkpoint paths now differ between `time_budget_mode=train` and `time_budget_mode=wall`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke --time-budget-mode wall --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python -m autoresearch_mlx.train --resume-from /tmp/autoresearch_budget_mode_resume --time-budget 1.5 --time-budget-mode wall --no-checkpoint`
- Measurements:
  - Budget-mode smoke summary:
    | Run shape                     | `time_budget_mode` | `training_seconds` | `budget_elapsed_seconds` | `total_seconds` | `session_steps` |
    | ----------------------------- | -------------------- | -------------------: | -------------------------: | ----------------: | ----------------: |
    | smoke default                 | `train`            |              `1.0` |                    `1.0` |           `1.1` |            `92` |
    | smoke no-checkpoint benchmark | `wall`             |              `1.0` |                    `1.0` |           `1.0` |            `91` |
    | resume with wall override     | `wall`             |              `1.5` |                    `1.5` |           `1.5` |           `159` |
- Interpretation:
  - The meaning of this change is not “make wall clock the new objective.” It is to make the objective explicit. `train` mode remains the core research default, while `wall` mode is now available when reduced elapsed blocking is itself the thing being measured.
  - The wall-mode resume check confirms that the new mode is a real runtime override, not just a fresh-run flag, while the distinct auto checkpoint slugs keep train-budget and wall-budget runs from clobbering each other.

### March 9, 2026 — `b68b750` — checkpoints: Benchmark checkpoint convergence and keep sync default — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested that checkpoint semantics be judged by convergence, not just save or resume latency, specifically by resuming halfway through training and comparing the end states.
  - Added a dedicated convergence harness for uninterrupted vs exact-resume vs `weights_only` midpoint resumes under a fixed optimizer-step budget.
  - Updated the repo guidance so future checkpoint semantic changes are expected to use this kind of end-state comparison.
  - Carried the checkpoint wrap-up through to an explicit policy conclusion: exact sync remains the default checkpoint mode even after testing async exact.

**Grounding**

- Files:
  - `tools/profile_resume_convergence.py`
  - `README.md`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile tools/profile_resume_convergence.py`
  - `./.venv/bin/python tools/profile_resume_convergence.py --preset m5-large --total-steps 80 --checkpoint-step 40 --eval-tokens 4096 --json-out results/analysis/m5_large_resume_convergence.json`
  - `./.venv/bin/python tools/profile_resume_convergence.py --preset m5-balanced --total-steps 200 --checkpoint-step 100 --eval-tokens 4096 --json-out results/analysis/m5_balanced_resume_convergence.json`
  - `./.venv/bin/python - <<'PY' ... > results/analysis/uninterrupted_repeatability.json`
- Measurements:
  - Midpoint resume-convergence comparison:

    | Preset          | Trajectory                       |   final loss | trailing loss | canonical `val_bpb` | relative RMS param drift vs uninterrupted |
    | --------------- | -------------------------------- | -----------: | ------------: | --------------------: | ----------------------------------------: |
    | `m5-balanced` | uninterrupted                    | `5.953065` |  `5.795487` |          `2.088155` |                             `0.000e+00` |
    | `m5-balanced` | exact midpoint resume            | `5.945498` |  `5.783511` |          `2.088881` |                             `5.009e-01` |
    | `m5-balanced` | `weights_only` midpoint resume | `6.139996` |  `5.601696` |          `2.194967` |                             `7.852e-01` |
    | `m5-large`    | uninterrupted                    | `5.846727` |  `6.018619` |          `2.188738` |                             `0.000e+00` |
    | `m5-large`    | exact midpoint resume            | `5.846764` |  `6.019718` |          `2.188925` |                             `2.616e-01` |
    | `m5-large`    | `weights_only` midpoint resume | `6.425774` |  `5.762017` |          `2.318585` |                             `6.785e-01` |
  - Uninterrupted same-seed repeatability baseline ([artifact](/Users/ent/Codex/autoresearch-everywhere/results/analysis/uninterrupted_repeatability.json)):

    | Preset          |   steps | run 1 canonical `val_bpb` | run 2 canonical `val_bpb` |         delta | relative RMS param drift |
    | --------------- | ------: | --------------------------: | --------------------------: | ------------: | -----------------------: |
    | `m5-balanced` | `200` |                `2.089673` |                `2.093795` | `+0.004122` |            `5.794e-01` |
    | `m5-large`    |  `80` |                `2.189574` |                `2.188915` | `-0.000659` |            `2.484e-01` |
  - Interpretation:

    - `weights_only` is clearly not trajectory-equivalent: on both tested presets it finishes at a worse canonical `val_bpb` and a materially different parameter state after the midpoint resume.
    - This is enough to treat `weights_only` as a failed experiment, not an active checkpoint direction. It remains useful as historical evidence and for targeted ablations, but it should no longer be recommended as the practical cheap-resume path.
    - Uninterrupted same-seed runs are not bitwise repeatable on this MLX stack either. On both tested presets, exact midpoint resume stays within the same broad parameter-drift envelope as uninterrupted-repeat baselines while keeping canonical `val_bpb` very close.
    - So the current evidence does not justify calling exact resume a standalone checkpoint-correctness bug. The tighter conclusion is that exact resume is metric-stable but not parameter-identical, which matches the underlying trainer's existing nondeterministic behavior on this machine.
    - This is the benchmark the checkpoint work was missing. Save cost and resume-ready latency tell you whether a checkpoint is cheap; they do not tell you whether it preserves the training trajectory.

### March 9, 2026 — `8359b0c` — checkpoints: Add async exact checkpoint writes — score `4` — complexity `8`

**Human-directed, AI-shaped (4)**

- Requested a pivot away from the failed `weights_only` path and toward async exact checkpointing instead.
  - Added an optional async exact write path that captures an exact host-side step-boundary snapshot on the training thread, then writes it in a background worker.
  - Kept the semantic target as exact resume rather than changing the resume contract again.
  - Updated the top-level docs to stop recommending `weights_only` and to frame it as a failed experiment retained only for comparison.
  - Tried a follow-on helper-process writer experiment with background process priority, but backed it out after ABAB runs showed it performed worse than the simpler thread-based async path.

**Grounding**

- Files:
  - `autoresearch_mlx/checkpoints.py`
  - `autoresearch_mlx/checkpoint_policy.py`
  - `autoresearch_mlx/train.py`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/checkpoints.py autoresearch_mlx/checkpoint_policy.py`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke --checkpoint-save-mode async --checkpoint-path /tmp/autoresearch_async_smoke --checkpoint-interval 0.5`
  - `./.venv/bin/python -m autoresearch_mlx.train --resume-from /tmp/autoresearch_async_smoke --time-budget 1.5 --checkpoint-save-mode async`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-large --time-budget 20 --benchmark-skip-eval --checkpoint-mode exact --checkpoint-save-mode sync --checkpoint-path /tmp/autoresearch_exact_sync_large --checkpoint-interval 2`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-large --time-budget 20 --benchmark-skip-eval --checkpoint-mode exact --checkpoint-save-mode async --checkpoint-path /tmp/autoresearch_exact_async_large --checkpoint-interval 2`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 60 --benchmark-skip-eval --checkpoint-mode exact --checkpoint-save-mode sync --checkpoint-path /tmp/autoresearch_exact_sync_xlarge --checkpoint-interval 2`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 60 --benchmark-skip-eval --checkpoint-mode exact --checkpoint-save-mode async --checkpoint-path /tmp/autoresearch_exact_async_xlarge --checkpoint-interval 2`
  - `./.venv/bin/python - <<'PY' ... > results/analysis/exact_async_abab_summary.json`
  - `./.venv/bin/python - <<'PY' ... > results/analysis/exact_async_process_abab_summary.json`
  - `./.venv/bin/python - <<'PY' ... > results/analysis/exact_async_wallclock_60s.json`
- Measurements:
  - Exact sync vs async checkpoint benchmark ([artifact](/Users/ent/Codex/autoresearch-everywhere/results/analysis/exact_async_checkpoint_benchmark.json)):

    | Preset        | Save mode | total seconds | blocking checkpoint % | total checkpoint write % | session tokens (M) | session steps | steady-state tok/s |
    | ------------- | --------- | ------------: | --------------------: | -----------------------: | -----------------: | ------------: | -----------------: |
    | `m5-large`  | sync      |      `20.5` |              `2.03` |                 `2.03` |          `0.258` |        `63` |        `12805.5` |
    | `m5-large`  | async     |      `20.4` |              `1.17` |                 `2.66` |          `0.254` |        `62` |        `12637.7` |
    | `m5-xlarge` | sync      |      `62.2` |              `3.37` |                 `3.37` |          `0.393` |        `96` |         `6556.8` |
    | `m5-xlarge` | async     |      `61.8` |              `1.66` |                 `6.09` |          `0.520` |       `127` |         `8581.8` |
  - Exact sync vs async ABAB benchmark ([artifact](/Users/ent/Codex/autoresearch-everywhere/results/analysis/exact_async_abab_summary.json)):

    | Preset        | Save mode | mean total seconds | mean blocking checkpoint % | mean total checkpoint write % | mean session tokens (M) | mean steady-state tok/s |
    | ------------- | --------- | -----------------: | -------------------------: | ----------------------------: | ----------------------: | ----------------------: |
    | `m5-large`  | sync      |           `20.5` |                   `1.93` |                      `1.93` |               `0.264` |             `13184.4` |
    | `m5-large`  | async     |           `20.3` |                   `1.07` |                      `3.03` |               `0.262` |             `13080.9` |
    | `m5-xlarge` | sync      |           `62.7` |                   `3.47` |                      `3.47` |               `0.393` |              `6309.7` |
    | `m5-xlarge` | async     |           `61.1` |                   `1.33` |                      `6.53` |               `0.381` |              `6329.5` |
  - Helper-process async exact ABAB benchmark ([artifact](/Users/ent/Codex/autoresearch-everywhere/results/analysis/exact_async_process_abab_summary.json)):

    | Preset        | Save mode            | mean total seconds | mean blocking checkpoint % | mean total checkpoint write % | mean session tokens (M) | mean steady-state tok/s |
    | ------------- | -------------------- | -----------------: | -------------------------: | ----------------------------: | ----------------------: | ----------------------: |
    | `m5-large`  | sync                 |           `20.7` |                   `2.20` |                      `2.20` |               `0.258` |             `12821.1` |
    | `m5-large`  | async helper process |           `20.7` |                   `2.02` |                      `7.73` |               `0.254` |             `12634.3` |
    | `m5-xlarge` | sync                 |           `62.3` |                   `3.41` |                      `3.41` |               `0.389` |              `6433.5` |
    | `m5-xlarge` | async helper process |           `62.2` |                   `2.45` |                     `15.27` |               `0.387` |              `6402.9` |
  - Exact sync vs async under a `60s` wall-clock budget ([artifact](/Users/ent/Codex/autoresearch-everywhere/results/analysis/exact_async_wallclock_60s.json)):

    | Preset        | Save mode | steps by cutoff | tokens by cutoff (M) |  wall tok/s | blocking checkpoint % of wall | total checkpoint write % of wall |
    | ------------- | --------- | --------------: | -------------------: | ----------: | ----------------------------: | -------------------------------: |
    | `m5-large`  | sync      |         `177` |            `0.725` | `12015.7` |                      `2.33` |                         `2.33` |
    | `m5-large`  | async     |         `180` |            `0.737` | `12255.8` |                      `0.66` |                         `3.37` |
    | `m5-xlarge` | sync      |         `111` |            `0.455` |  `7508.4` |                      `3.88` |                         `3.88` |
    | `m5-xlarge` | async     |         `114` |            `0.467` |  `7762.6` |                      `1.13` |                         `5.91` |
  - Interpretation:

    - Async exact is viable because it keeps the resume semantics exact while moving the expensive on-disk write out of the training thread. The host snapshot path preserves `bfloat16` tensors exactly by reinterpreting them as `uint16` during transfer and viewing them back on restore.
    - Under the repo's usual training-time budget, the grounded result is still mixed: async exact reliably lowers blocking checkpoint time, but the matched ABAB runs do not show a stable end-to-end throughput win.
    - Under a real `60s` wall-clock budget, async exact does help on both tested heavier presets, because lowering blocking time lets the run complete slightly more optimizer steps before the deadline.
    - That is useful, but it is not enough to flip the default. The current policy conclusion is still to keep synchronous exact checkpoints as the default path, and to treat async exact as an optional wall-clock-oriented variant.
    - The helper-process follow-up did not help. Running the writer in a forked background process with lower process priority reduced blocking a little, but increased total write share sharply and failed to improve throughput on either preset.
    - That points away from scheduler isolation as the main bottleneck. The likely limit is still the cost of capturing and moving the exact host snapshot through shared memory, so the simpler thread-based async writer remains the better current implementation.

### March 9, 2026 — `fd5358b` — checkpoints: Make auto checkpoint cadence mode-aware — score `2`

**AI-identified within brief, human-approved (2)**

- Followed the new checkpoint-mode work by teaching the automatic cadence selector and tradeoff tool about `weights_only` as its own measured profile instead of making it inherit the exact full-state interval.
  - Replaced the earlier provisional `weights_only` save-cost estimates with stronger repeated-save matched-run measurements on `m5-large` and `m5-xlarge`.
  - Updated the runtime selector, runtime policy message, and tradeoff analysis so exact and `weights_only` now produce different default intervals on the calibrated M5 shapes.

**Grounding**

- Files:
  - `README.md`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_mlx/checkpoint_policy.py`
  - `tools/checkpoint_tradeoff.py`
  - `autoresearch_mlx/train.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/checkpoint_policy.py tools/checkpoint_tradeoff.py autoresearch_mlx/train.py`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-large --time-budget 20 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-large --time-budget 20 --benchmark-skip-eval --checkpoint-mode weights_only --checkpoint-path /tmp/autoresearch_m5_large_weights_only --checkpoint-interval 2`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 60 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 60 --benchmark-skip-eval --checkpoint-mode weights_only --checkpoint-path /tmp/autoresearch_m5_xlarge_weights_only --checkpoint-interval 2`
  - Verified via `.venv` Python snippet that `choose_auto_checkpoint_decision(...)` now resolves exact to `2m` and `weights_only` to `1m` on both calibrated parameter bands.
  - Verified via `.venv` Python snippet that `resolve_checkpoint_settings(...)` prints the same mode-aware `2m` vs `1m` result for `m5-large`, including distinct auto checkpoint paths per mode.
  - `env PYTHONPATH=/Users/ent/Codex/autoresearch-everywhere ./.venv/bin/python tools/checkpoint_tradeoff.py`
- Measurements:
  - Repeated-save `weights_only` overhead from matched long-window runs:

    | Preset        | time budget (s) |  saves | non-training overhead baseline (s) | non-training overhead with `weights_only` (s) | added wall overhead (s) | approx. save cost (ms/save) |
    | ------------- | --------------: | -----: | ---------------------------------: | ----------------------------------------------: | ----------------------: | --------------------------: |
    | `m5-large`  |          `20` | `10` |                            `0.0` |                                         `0.2` |                 `0.2` |                      `20` |
    | `m5-xlarge` |          `60` | `28` |                            `0.1` |                                         `0.9` |                 `0.8` |                      `29` |
  - Mode-aware default interval outcomes on the calibrated M5 profiles:

    | Preset band                    | exact default                            | `weights_only` default                 |
    | ------------------------------ | ---------------------------------------- | ---------------------------------------- |
    | up to `m5-large`             | `2m` at `0.0583%` save-only overhead | `1m` at `0.0333%` save-only overhead |
    | `m5-xlarge` / upstream-scale | `2m` at `0.0917%` save-only overhead | `1m` at `0.0500%` save-only overhead |
  - Interpretation:

    - The stronger repeated-save measurements confirm that `weights_only` is cheap enough to justify a shorter default interval than exact full-state resume under the same fixed-overhead cap.
    - Exact remains at `2m` because its measured save cost still pushes `1m` above the current `0.1%` save-only overhead target on the calibrated M5 shapes.
    - `weights_only` now resolves to `1m` because its repeated-save overhead is materially lower, while exact still remains the default when exact optimizer/loader continuity matters.

### March 9, 2026 — `1b6887f` — checkpoints: Add approximate weights-only checkpoint mode — score `4` — complexity `8`

**Human-directed, AI-shaped (4)**

- Requested checkpoint semantic optimization, while leaving the concrete mechanism to the agent.
  - Added an explicit `weights_only` checkpoint mode alongside exact full-state resume.
  - Threaded the new mode through the trainer, checkpoint metadata, auto checkpoint path selection, and both checkpoint profiling tools.
  - Kept exact full-state resume as the default, and kept the auto cadence conservatively calibrated from the existing exact-resume measurements.
  - Semantically, `weights_only` saves model weights plus checkpoint metadata, intentionally omits optimizer and train-loader state, deletes stale exact-state payloads in the target directory, and resumes approximately from a fresh optimizer and loader stream.

**Grounding**

- Files:
  - `README.md`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
  - `autoresearch_mlx/checkpoint_policy.py`
  - `autoresearch_mlx/checkpoints.py`
  - `tools/checkpoint_tradeoff.py`
  - `tools/profile_checkpoint_path.py`
  - `tools/profile_resume_ready.py`
  - `autoresearch_mlx/train.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/checkpoint_policy.py autoresearch_mlx/checkpoints.py tools/checkpoint_tradeoff.py tools/profile_checkpoint_path.py tools/profile_resume_ready.py autoresearch_mlx/train.py`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke --checkpoint-mode weights_only --checkpoint-path /tmp/autoresearch_weights_only_smoke --checkpoint-interval 0.5`
  - `./.venv/bin/python -m autoresearch_mlx.train --resume-from /tmp/autoresearch_weights_only_smoke --time-budget 1.5 --checkpoint-mode weights_only`
  - `./.venv/bin/python tools/profile_resume_ready.py --preset m5-large --checkpoint-mode exact --resume-steps 2 --repeats 3 --json-out results/analysis/m5_large_exact_resume_ready_v2.json`
  - `./.venv/bin/python tools/profile_resume_ready.py --preset m5-large --checkpoint-mode weights_only --resume-steps 2 --repeats 3 --json-out results/analysis/m5_large_weights_only_resume_ready_v2.json`
  - `./.venv/bin/python tools/profile_resume_ready.py --preset m5-xlarge --checkpoint-mode exact --resume-steps 2 --repeats 3 --json-out results/analysis/m5_xlarge_exact_resume_ready_v2.json`
  - `./.venv/bin/python tools/profile_resume_ready.py --preset m5-xlarge --checkpoint-mode weights_only --resume-steps 2 --repeats 3 --json-out results/analysis/m5_xlarge_weights_only_resume_ready_v2.json`
  - `./.venv/bin/python tools/profile_checkpoint_path.py --preset m5-xlarge --checkpoint-mode exact --json-out results/analysis/xlarge_exact_checkpoint_profile.json`
  - `./.venv/bin/python tools/profile_checkpoint_path.py --preset m5-xlarge --checkpoint-mode weights_only --json-out results/analysis/xlarge_weights_only_checkpoint_profile.json`
- Measurements:
  - Exact vs `weights_only` resume-ready medians (`3` trials each, prepacked path, `5` seed train steps before save):

    | Preset        | Mode             | median save (s) | median restore (s) | median resume-ready (s) | median first resumed step wall (s) |
    | ------------- | ---------------- | --------------: | -----------------: | ----------------------: | ---------------------------------: |
    | `m5-large`  | `exact`        |       `0.057` |          `0.017` |               `1.129` |                          `1.033` |
    | `m5-large`  | `weights_only` |       `0.023` |          `0.010` |               `1.176` |                          `1.104` |
    | `m5-xlarge` | `exact`        |       `0.084` |          `0.028` |               `1.215` |                          `1.125` |
    | `m5-xlarge` | `weights_only` |       `0.042` |          `0.015` |               `1.236` |                          `1.152` |
  - Direct `m5-xlarge` save-path attribution:

    | Mode             | save total (s) | model bytes | optimizer bytes | loader bytes |
    | ---------------- | -------------: | ----------: | --------------: | -----------: |
    | `exact`        |      `0.497` |  `159.4M` |      `218.3M` |       `22` |
    | `weights_only` |      `0.432` |  `159.4M` |           `0` |        `0` |
  - Interpretation:

    - `weights_only` materially reduces save cost: about `-60%` on `m5-large` median save time and about `-50%` on `m5-xlarge`.
    - Resume-ready latency does not improve in the current measurements. The first resumed optimizer step still dominates the path back to productive training, and `weights_only` resumes approximately with a fresh optimizer and train-loader state.
    - The direct `m5-xlarge` save-path profile explains why the save win is bounded: omitting optimizer state removes about `218 MB` of writes, but the model weights still dominate the checkpoint payload.
    - The meaning and purpose of the mode are operational rather than numerical: it is a cheaper freshness checkpoint for long local runs when exact optimizer/loader continuity is not worth the extra write cost.
    - This makes `weights_only` a useful cheaper approximate snapshot mode, not a replacement for exact step-boundary resume when continuity matters.

### March 9, 2026 — `39c9055` — checkpoints: Calibrate tradeoff analysis with resume-ready penalty — score `2`

**AI-identified within brief, human-approved (2)**

- Threaded the new measured resume-ready medians into the checkpoint calibration and tradeoff analysis so the tooling stops treating resume penalty as zero.
  - Kept the runtime selector itself save-overhead-based for now; this change is about analysis fidelity, not changing the default interval policy yet.
  - Updated the runtime checkpoint-policy log line to surface the measured resume-ready penalty for the selected calibration.

**Grounding**

- Files:
  - `autoresearch_mlx/checkpoint_policy.py`
  - `tools/checkpoint_tradeoff.py`
  - `autoresearch_mlx/train.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/checkpoint_policy.py tools/checkpoint_tradeoff.py autoresearch_mlx/train.py`
  - `env PYTHONPATH=/Users/ent/Codex/autoresearch-everywhere ./.venv/bin/python tools/checkpoint_tradeoff.py`
- Measurements:
  - Updated scenario table with measured resume-ready penalties and a clean split between fixed save overhead and projected total waste:

    | Profile                        | Events/day | Optimal interval (min) | Fixed checkpoint overhead (%) | Projected total waste (%) | Save cost (ms) | Resume penalty (ms) |
    | ------------------------------ | ---------: | ---------------------: | ----------------------------: | ------------------------: | -------------: | ------------------: |
    | `m5-large` exact full-state  |      `1` |               `1.83` |                     `0.064` |                 `0.128` |         `70` |             `641` |
    | `m5-xlarge` exact full-state |      `1` |               `2.30` |                     `0.080` |                 `0.160` |        `110` |             `683` |
  - Interpretation:

    - The fixed checkpoint overhead column is now the actual save-time tax from taking checkpoints at the chosen interval. It does not depend on the assumed restart frequency.
    - The projected total waste column is the modeled all-in waste under the assumed resume-needed event rate, including save overhead, lost work, and measured resume-ready penalty.
    - Young/Daly-style optimal intervals are unchanged because the resume penalty is interval-independent, but the projected total waste curves are now more honest about restart cost.
    - At realistic interruption rates, the measured resume penalty is not large enough to overturn the current human-factors `2m` recommendation, but it does materially raise the modeled waste percentage for frequent-resume scenarios.
    - This keeps the tradeoff tooling aligned with the stronger resume-ready benchmark without silently changing the runtime auto-checkpoint behavior.

### March 9, 2026 — `07a0707` — checkpoints: Benchmark resume-ready checkpoint latency — score `2`

**AI-identified within brief, human-approved (2)**

- Surfaced a stronger grounding need for checkpoint optimization: raw `mx.load` timings are not enough, because they understate what a resumed run actually pays before it becomes productive again.
  - Added a dedicated resume-ready benchmark that measures cold runtime build, checkpoint restore, and the first completed optimizer step after resume.
  - Added repeat support and aggregate summaries so checkpoint policy work can use medians instead of hanging on a single noisy resumed run.

**Grounding**

- Files:
  - `tools/profile_resume_ready.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile tools/profile_resume_ready.py`
  - `./.venv/bin/python tools/profile_resume_ready.py --preset m5-large --resume-steps 2 --repeats 3 --json-out results/analysis/m5_large_resume_ready.json`
  - `./.venv/bin/python tools/profile_resume_ready.py --preset m5-xlarge --resume-steps 2 --repeats 3 --json-out results/analysis/m5_xlarge_resume_ready.json`
- Measurements:
  - Resume-ready summary (`3` trials each, prepacked path, `5` seed train steps before save):

    | Preset        | median resume-ready (s) | median runtime build (s) | median restore (s) | median first step wall (s) | median steady step wall (s) | median resume tax vs steady wall (s) |
    | ------------- | ----------------------: | -----------------------: | -----------------: | -------------------------: | --------------------------: | -----------------------------------: |
    | `m5-large`  |               `0.641` |                `0.039` |          `0.016` |                  `0.581` |                   `0.568` |                            `0.073` |
    | `m5-xlarge` |               `0.683` |                `0.060` |          `0.020` |                  `0.549` |                   `0.538` |                            `0.168` |
  - Interpretation:

    - The user-facing resume-ready delay is not dominated by checkpoint metadata or loader restore. It is mostly runtime rebuild plus the first resumed optimizer step.
    - `m5-large` is reasonably tight across the three trials; the median resume-ready delay is about `0.64s`, with about `73ms` above a steady resumed step.
    - `m5-xlarge` is centered closer to about `0.68s`, and one of the three trials again had a much slower first resumed step. That makes the median much more trustworthy than the mean for policy work at this shape.
    - This is the right benchmark to use for future checkpoint-mode comparisons, because it measures "time until the resumed run is productive again" directly instead of inferring from lazy file-load timings.

### March 9, 2026 — `26e4c64` — train: Auto-detect benchmark warmup cutoff — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested that the benchmark warmup cutoff be determined statistically by default instead of relying on a manually specified fixed step count, and that the resulting cutoff be tracked explicitly as `warmup_done_step`.
  - Switched the benchmark accounting to record per-step session timings and determine the warmup boundary after the run.
  - Preserved `--benchmark-warmup-steps` as an explicit fixed override for ablations and replay.
  - Added `benchmark_warmup_mode` and `warmup_done_step` to the training summary so the cutoff is explicit in the reported metrics.

**Grounding**

- Files:
  - `autoresearch_mlx/train.py`
  - `docs/program-mlx.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-fast --time-budget 5 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-fast --time-budget 5 --benchmark-warmup-steps 62 --benchmark-skip-eval --no-checkpoint`
- Measurements:
  - Detection rule:

    - The auto cutoff compares early-step windows against a trailing reference window taken from the end of the run instead of guessing a fixed warmup length ahead of time.
    - The reference center is the trailing-window median, and the tolerance is the larger of a `3%` relative band or `3 x` the trailing-window median absolute deviation.
    - The chosen cutoff is the first step prefix after which a short stability window stays within that tolerance, so `warmup_done_step` marks the first step whose timing looks statistically indistinguishable from the trailing steady-state band.
  - Auto-vs-fixed `m5-fast` comparison (`5s`, eval skipped, no checkpoint):

    | Mode      | `benchmark_warmup_steps` | `warmup_done_step` | warmup step seconds | warmup wall seconds | steady-state steps | steady-state training seconds | steady-state `tok_per_sec` |
    | --------- | -------------------------: | -------------------: | ------------------: | ------------------: | -----------------: | ----------------------------: | ---------------------------: |
    | `auto`  |                     `37` |               `37` |           `1.258` |           `1.291` |            `316` |                     `3.749` |                  `43153.8` |
    | `fixed` |                     `62` |               `62` |           `1.012` |           `1.046` |            `333` |                     `3.990` |                  `42734.9` |
  - Interpretation:

    - The default statistical detector produces a concrete benchmark cutoff without requiring a manually chosen step count.
    - The detector is responsive to real startup shape, so it should not be expected to pick the same cutoff across materially different warmup profiles.
    - On the final `m5-fast` rerun, the auto and fixed modes still landed in the same steady-state performance band, which is the main practical requirement: the new default is more ergonomic without obscuring the benchmark semantics.

### March 9, 2026 — `9330302` — checkpoints: Profile checkpoint save/load path — score `2`

**AI-identified within brief, human-approved (2)**

- Surfaced checkpoint persistence as the next optimization target now that runtime instrumentation, the prepacked fast path, and the live-packing fallback have all been tightened.
  - Added a dedicated save/load profiler so checkpoint work can be aimed at the dominant cost center instead of guessing from end-to-end wall time alone.

**Grounding**

- Files:
  - `tools/profile_checkpoint_path.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile tools/profile_checkpoint_path.py`
  - `./.venv/bin/python tools/profile_checkpoint_path.py --preset m5-large --train-steps 5`
  - `./.venv/bin/python tools/profile_checkpoint_path.py --preset m5-xlarge --train-steps 5`
- Measurements:
  - Save-path breakdown on the default prepared prepacked path:

    | Preset        | save total (s) | model write (s) | optimizer write (s) | loader write (s) | metadata write (s) | model bytes | optimizer bytes | loader bytes |
    | ------------- | -------------: | --------------: | ------------------: | ---------------: | -----------------: | ----------: | --------------: | -----------: |
    | `m5-large`  |      `0.254` |       `0.224` |           `0.028` |       `0.0017` |         `0.0008` |   `80.2M` |      `118.1M` |       `22` |
    | `m5-xlarge` |      `0.473` |       `0.422` |           `0.050` |       `0.0004` |         `0.0002` |  `159.4M` |      `218.3M` |       `22` |
  - Interpretation:

    - On the shipped prepacked path, checkpoint cost is overwhelmingly dominated by writing model and optimizer safetensors. Loader serialization and metadata are effectively free by comparison.
    - That means the next meaningful checkpoint optimization is not Python-side cleanup inside the current save path; it would need to change semantics or scheduling, such as lighter resume tiers or asynchronous/background save behavior.
    - The current restore-side numbers from this tool are exploratory only. `mx.load` appears lazy enough that raw file-load timings understate "resume ready" cost, so restore optimization should not be driven from those numbers yet without a stronger resume-readiness benchmark.

### March 9, 2026 — `9a79473` — data: Optimize live-packed fallback buffer — score `2`

**AI-identified within brief, human-approved (2)**

- Surfaced the live-packing fallback as the next optimization target after the shipped preset fast path and warmup-aware benchmark tooling were in place.
  - Replaced the linear best-fit scan with a length-indexed FIFO packing buffer for the live-packed training path.
  - Reused the same packing structure in prepacked-cache construction so the packing behavior stays consistent across both paths.

**Grounding**

- Files:
  - `autoresearch_mlx/data.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/data.py autoresearch_mlx/train.py tools/profile_loader_path.py`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-balanced --steps 80 --warmup-steps 5 --no-prepacked-cache`
  - `./.venv/bin/python /tmp/autoresearch_livepack_before/tools/profile_loader_path.py --preset m5-balanced --steps 80 --warmup-steps 5 --no-prepacked-cache`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-large --steps 80 --warmup-steps 5 --no-prepacked-cache`
  - `./.venv/bin/python /tmp/autoresearch_livepack_before/tools/profile_loader_path.py --preset m5-large --steps 80 --warmup-steps 5 --no-prepacked-cache`
- Measurements:
  - Matched fallback-only before/after profile (`80` measured steps after `5` warmup steps):

    | Preset          | version | `tok_per_sec` | total mean (ms) | loader mean (ms) | grad mean (ms) | optimizer mean (ms) |
    | --------------- | ------- | --------------: | --------------: | ---------------: | -------------: | ------------------: |
    | `m5-balanced` | before  |    `33306.25` |       `61.49` |        `0.963` |      `45.85` |           `13.47` |
    | `m5-balanced` | after   |    `34334.42` |       `59.65` |        `0.174` |      `45.14` |           `13.28` |
    | `m5-large`    | before  |    `14456.33` |      `283.34` |        `1.328` |     `222.57` |           `52.66` |
    | `m5-large`    | after   |    `14522.86` |      `282.04` |        `0.245` |     `222.67` |           `52.40` |
  - Interpretation:

    - The new packing buffer removes most of the Python-side fallback cost on both tested presets: loader-call time dropped by about `82%` on `m5-balanced` and `m5-large`.
    - The end-to-end win is real but modest because these steps are still compute-dominated: about `+3.1% tok/s` on `m5-balanced` and `+0.5% tok/s` on `m5-large`.
    - This is worth keeping as a fallback-path cleanup, but the grounded effect is much smaller than the raw loader-time drop might suggest.

### March 9, 2026 — `8c6ed4f` — train: Add warmup-aware MLX benchmark tooling — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested that we understand the apparent `m5-xlarge` prepacked regression rather than guess at it.
  - Added a focused loader-path profiling harness that separates `next(loader)`, forced `mx.eval(x, y)`, gradient computation, and optimizer update timing.
  - Added warmup-aware benchmark reporting to `autoresearch_mlx/train.py`, including startup/warmup seconds and steady-state tokens-per-second, with an option to skip eval noise during benchmarking.
  - Used it to test whether the observed xlarge regression was actually in the prepacked loader path or was just run-level noise.

**Grounding**

- Files:
  - `autoresearch_mlx/train.py`
  - `tools/profile_loader_path.py`
  - `docs/program-mlx.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py tools/profile_loader_path.py`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke --benchmark-warmup-steps 1 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 60 --benchmark-warmup-steps 5 --benchmark-skip-eval --no-checkpoint`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 60 --benchmark-warmup-steps 5 --benchmark-skip-eval --no-checkpoint --no-prepacked-cache`
  - repeated `ABAB` run using the same two commands above
  - `python3 -m py_compile tools/profile_loader_path.py`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-xlarge --steps 80 --warmup-steps 5`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-xlarge --steps 80 --warmup-steps 5 --no-prepacked-cache`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-xlarge --steps 20 --warmup-steps 5`
  - `./.venv/bin/python tools/profile_loader_path.py --preset m5-xlarge --steps 20 --warmup-steps 5 --no-prepacked-cache`
- Measurements:
  - `80` measured xlarge steps after `5` warmup steps:

    | Train path                 | `tok_per_sec` | total mean / median (ms) | loader mean / median (ms) | grad mean / median (ms) | optimizer mean / median (ms) |
    | -------------------------- | --------------: | -----------------------: | ------------------------: | ----------------------: | ---------------------------: |
    | prepacked cache            |     `7637.86` |      `536.28 / 534.48` |           `0.46 / 0.08` |     `447.99 / 446.38` |            `84.05 / 83.95` |
    | token cache / live packing |     `7359.01` |      `556.60 / 549.47` |           `1.66 / 0.90` |     `463.84 / 456.97` |            `87.54 / 86.47` |
  - `20` measured xlarge steps after `5` warmup steps:

    | Train path                 | steady `tok_per_sec` | steady total mean / median (ms) | warmup total mean / median (ms) |
    | -------------------------- | ---------------------: | ------------------------------: | ------------------------------: |
    | prepacked cache            |            `7742.33` |             `529.04 / 527.70` |             `619.00 / 541.36` |
    | token cache / live packing |            `7736.51` |             `529.44 / 527.13` |             `554.78 / 539.72` |
  - Interpretation:

    - There is no evidence here of a structural steady-state xlarge regression caused by the prepacked loader path.
    - The only consistent loader-path effect is that prepacked reduces `loader_call` time. Forced `mx.eval(x, y)` materialization stays effectively zero on both paths, so the earlier regression is not explained by unified-memory handoff showing up outside the loader bucket.
    - The earlier `60s` end-to-end xlarge regression now looks more like run-level variance or warmup/compile noise than a genuine fast-path loss.
  - Warmup-aware `ABAB` end-to-end benchmark (`60s`, `benchmark_warmup_steps=5`, eval skipped):

    | Leg    | Train path                 | `session_steps` | `session_tokens_M` | warmup step seconds | steady-state `tok_per_sec` | `train_tflops` | `loader_percent` |
    | ------ | -------------------------- | ----------------: | -------------------: | ------------------: | ---------------------------: | ---------------: | -----------------: |
    | `A1` | prepacked cache            |           `116` |            `0.475` |           `3.229` |                   `7974.2` |        `2.206` |           `0.04` |
    | `B1` | token cache / live packing |           `115` |            `0.471` |           `2.686` |                   `7858.0` |        `2.176` |           `0.27` |
    | `A2` | prepacked cache            |           `116` |            `0.475` |           `3.101` |                   `7938.6` |        `2.197` |           `0.06` |
    | `B2` | token cache / live packing |           `115` |            `0.471` |           `2.738` |                   `7809.5` |        `2.163` |           `0.38` |
  - Interpreting the warmup-aware `ABAB` run:

    - Prepacked pays an extra startup cost of about `0.45s` on average across the first `5` warmup steps.
    - After warmup, prepacked runs about `1.6%` faster in steady-state (`7956 tok/s` average vs `7834 tok/s` average).
    - That implies a rough xlarge crossover at about `56` steady-state steps, or about `29s`, before the prepacked path amortizes its slower warmup and comes out ahead overall.
    - Under the repo's current `60s` benchmark window, the xlarge prepacked path should now be treated as a small net win, not a regression.

### March 9, 2026 — `d7f4d23` — data: Make prepacked caches default for shipped presets — score `2`

**AI-identified within brief, human-approved (2)**

- Surfaced making the shipped M5 presets use prepacked caches as the normal prepared-state fast path instead of treating them as an extra opt-in, and the user refreshed that priority.
  - Switched `autoresearch_mlx/prepare.py` to build the shipped prepacked cache coverage by default.
  - Added an explicit opt-out path for intentionally leaving the live packing fallback in place.
  - Tightened the loader logging so missing prepacked coverage is visible instead of silently falling through to token-cache/live packing.

**Grounding**

- Files:
  - `autoresearch_mlx/prepare.py`
  - `autoresearch_mlx/data.py`
  - `README.md`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/prepare.py autoresearch_mlx/data.py`
  - `./.venv/bin/python -m autoresearch_mlx.prepare --num-shards 1`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-large --time-budget 0.2 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 0.2 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-large --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-large --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512 --no-prepacked-cache`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512 --no-prepacked-cache`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-balanced --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-balanced --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512 --no-prepacked-cache`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-fast --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-fast --time-budget 60 --eval-tokens 512 --canonical-eval-tokens 512 --no-prepacked-cache`
- Measurements:
  - Default `autoresearch_mlx/prepare.py --num-shards 1` behavior on the existing cache directory detected the raw data, tokenizer, and token caches, then built the missing shipped prepacked coverage automatically:

    - `train seq_len=1024`
    - `train seq_len=2048`
    - `val seq_len=1024`
    - `val seq_len=2048`
  - The shipped preset coverage now hits the prepacked train path without extra preparation flags:

    - `m5-fast`: `Data loader (train): using prepacked cache.`
    - `m5-balanced`: `Data loader (train): using prepacked cache.`
    - `m5-large`: `Data loader (train): using prepacked cache.`
    - `m5-xlarge`: `Data loader (train): using prepacked cache.`
  - `60s` A/B comparison (`512` proxy/canonical eval tokens):

    | Preset          | Train path                 |  `val_bpb` | `proxy_val_bpb` | `train_tflops` | `loader_percent` | `session_steps` | `session_tokens_M` |
    | --------------- | -------------------------- | -----------: | ----------------: | ---------------: | -----------------: | ----------------: | -------------------: |
    | `m5-fast`     | prepacked cache            | `1.956952` |      `1.920689` |        `0.624` |           `0.44` |          `7741` |            `3.963` |
    | `m5-fast`     | token cache / live packing | `1.975841` |      `1.901032` |        `0.594` |           `3.58` |          `7369` |            `3.773` |
    | `m5-balanced` | prepacked cache            | `1.730654` |      `1.739594` |        `1.384` |           `0.18` |          `1072` |            `2.195` |
    | `m5-balanced` | token cache / live packing | `1.731765` |      `1.707002` |        `1.306` |           `1.23` |          `1014` |            `2.077` |
    | `m5-large`    | prepacked cache            | `1.911791` |      `1.929047` |        `1.692` |           `0.13` |           `224` |            `0.918` |
    | `m5-large`    | token cache / live packing | `1.906361` |      `1.934655` |        `1.607` |           `0.51` |           `213` |            `0.872` |
    | `m5-xlarge`   | prepacked cache            | `2.169967` |      `2.132617` |        `2.152` |           `0.05` |           `114` |            `0.467` |
    | `m5-xlarge`   | token cache / live packing | `2.140543` |      `2.115287` |        `2.233` |           `0.36` |           `118` |            `0.483` |
  - Interpreting the matched runs:

    - `m5-fast` shows a clear prepacked win: `7741` vs `7369` steps, `3.963M` vs `3.773M` session tokens (`+5.0%`), and `loader_percent` dropped from `3.58` to `0.44`.
    - `m5-balanced` also shows a clear prepacked win: `1072` vs `1014` steps, `2.195M` vs `2.077M` session tokens (`+5.7%`), and `loader_percent` dropped from `1.23` to `0.18`.
    - `m5-large` shows a real fast-path win from prepacking: `224` vs `213` steps in the same `60s`, `0.918M` vs `0.872M` session tokens (`+5.3%`), and `loader_percent` dropped from `0.51` to `0.13`.
    - `m5-xlarge` showed lower `loader_percent` but a small single-run loss on total tokens; later dedicated profiling suggests that negative result was likely run-level variance rather than a structural steady-state fast-path regression.

### March 9, 2026 — `b2b08df` — train: Add robust MLX utilization instrumentation — score `4` — complexity `8`

**Human-directed, AI-shaped (4)**

- Requested that optimization work focus on robust utilization instrumentation rather than another raw speed change.
  - Replaced the fake per-step `mfu` readout with measured step compute-share utilization and estimated training TFLOPs.
  - Added persistent step-telemetry counters so resumed runs keep coherent step-level utilization summaries.
  - Added explicit loader/grad/accumulate/optimizer/checkpoint/eval breakdowns to the final run summary while retaining `mfu_percent` as a backward-compatible alias.
  - Tightened resumed-run reporting so current-invocation timing and checkpoint percentages no longer mix with cumulative training progress.
    - `training_seconds`, `total_seconds`, `checkpoint_percent`, `eval_percent`, and `checkpoint_count` are now current-invocation metrics.
    - `cumulative_training_seconds`, `cumulative_checkpoint_seconds`, and `cumulative_checkpoint_count` are printed separately.
    - utilization remains resume-aware because the step-telemetry window is restored from checkpoints.

**Grounding**

- Files:
  - `autoresearch_mlx/train.py`
  - `autoresearch_mlx/checkpoints.py`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/checkpoints.py`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-balanced --time-budget 20 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-fast --time-budget 20 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-large --time-budget 20 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 20 --eval-tokens 512 --canonical-eval-tokens 512`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke --checkpoint-path /tmp/autoresearch_util_resume_smoke --checkpoint-interval 0.5`
  - `./.venv/bin/python -m autoresearch_mlx.train --resume-from /tmp/autoresearch_util_resume_smoke --time-budget 1.5`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke --checkpoint-path /tmp/autoresearch_resume_metrics --checkpoint-interval 0.5`
  - `./.venv/bin/python -m autoresearch_mlx.train --resume-from /tmp/autoresearch_resume_metrics --time-budget 1.5`
- Measurements:
  - `20s` profile comparison runs (`512` proxy/canonical eval tokens):

    | Preset          |  `val_bpb` | `proxy_val_bpb` | `mfu_percent` | `train_tflops` | `loader_percent` | `grad_percent` | `accum_percent` | `optimizer_percent` | `other_step_percent` | `checkpoint_percent` | `eval_percent` | `peak_vram_mb` | `util_window_steps` | Train loader               |
    | --------------- | -----------: | ----------------: | --------------: | ---------------: | -----------------: | ---------------: | ----------------: | --------------------: | ---------------------: | ---------------------: | ---------------: | ---------------: | --------------------: | -------------------------- |
    | `m5-fast`     | `2.018503` |      `1.946999` |       `99.50` |        `0.706` |           `0.45` |        `66.24` |          `5.97` |             `27.30` |               `0.05` |               `0.00` |         `0.11` |        `146.5` |              `2912` | prepacked cache            |
    | `m5-balanced` | `1.929286` |      `1.908801` |       `99.83` |        `1.359` |           `0.16` |        `75.85` |          `1.77` |             `22.21` |               `0.01` |               `0.00` |         `0.17` |        `949.9` |               `349` | prepacked cache            |
    | `m5-large`    | `2.184426` |      `2.200882` |       `99.28` |        `1.716` |           `0.71` |        `78.84` |          `2.22` |             `18.23` |               `0.01` |               `0.00` |         `0.42` |       `1944.3` |                `74` | token cache / live packing |
    | `m5-xlarge`   | `2.246450` |      `2.284754` |       `99.86` |        `2.207` |           `0.13` |        `83.54` |          `0.67` |             `15.64` |               `0.00` |               `0.00` |         `0.78` |       `4294.2` |                `38` | token cache / live packing |
  - smoke run:

    - `mfu_percent=99.32`
    - `train_tflops=0.564`
    - `loader_percent=0.63`
    - `optimizer_percent=26.50`
  - smoke checkpoint/resume path:

    - checkpointed smoke ended with `checkpoint_count=2` and `checkpoint_percent=2.24`
    - resumed smoke ended with `mfu_percent=99.43`, `train_tflops=0.591`, `util_window_steps=175`, and `checkpoint_count=1`
  - resume-metrics smoke path:

    - checkpointed smoke ended with `training_seconds=1.0`, `total_seconds=1.1`, `checkpoint_count=2`, `cumulative_training_seconds=1.0`, `cumulative_checkpoint_seconds=0.020`, and `cumulative_checkpoint_count=2`
    - resumed smoke ended with `training_seconds=0.5`, `total_seconds=0.5`, `checkpoint_count=1`, `cumulative_training_seconds=1.5`, `cumulative_checkpoint_seconds=0.027`, and `cumulative_checkpoint_count=3`
- Confirmed behavior:
  - per-step progress lines now report `util` and `tflops` instead of a fake `mfu`
  - final summaries now expose the step-time split and end-of-run checkpoint/eval shares
  - resumed runs restore the saved step-telemetry state and continue the utilization window instead of resetting it to zero
  - resumed runs no longer print a cumulative `training_seconds` beside a per-invocation `total_seconds`
  - across the shipped M5 presets, utilization stays near saturation while `train_tflops` rises with model size and the larger presets still reveal the token-cache/live-pack fallback in their train-loader path
  - no performance-improvement claim is attached to this change; the grounding here is instrumentation correctness and observability

### March 9, 2026 — `62f2f9c` — checkpoints: Auto-enable checkpoint cadence for longer runs — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested that the checkpoint-frequency selector be wired into `autoresearch_mlx/train.py` as a default-on behavior for runs longer than 5 minutes.
  - Added a shared checkpoint-policy module and used it from the trainer so long runs now auto-select a checkpoint cadence from the measured save-cost calibrations.
  - Made `--checkpoint-path` keep the default cadence selector unless `--checkpoint-interval` is explicitly pinned.
  - Added `--no-checkpoint` as the explicit escape hatch for disabling the automatic long-run behavior.

**Grounding**

- Files:
  - `autoresearch_mlx/train.py`
  - `README.md`
  - `docs/program-mlx.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/checkpoint_policy.py tools/checkpoint_tradeoff.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - interrupted startup probe: `.venv/bin/python -m autoresearch_mlx.train --preset m5-balanced --time-budget 301 --eval-tokens 512 --canonical-eval-tokens 512`
  - interrupted startup probe: `.venv/bin/python -m autoresearch_mlx.train --preset m5-balanced --time-budget 301 --checkpoint-path /tmp/autoresearch-policy-existing-path --eval-tokens 512 --canonical-eval-tokens 512`
  - interrupted startup probe: `.venv/bin/python -m autoresearch_mlx.train --preset m5-balanced --time-budget 301 --no-checkpoint --eval-tokens 512 --canonical-eval-tokens 512`
  - interrupted behavior probe: `.venv/bin/python -m autoresearch_mlx.train --smoke --time-budget 10 --checkpoint-interval 0.5`
- Confirmed behavior:
  - `time_budget > 300s` with no checkpoint flags now auto-selects `checkpoint_interval=120.0` and an automatic checkpoint directory for `m5-balanced`.
  - `time_budget > 300s` with an explicit `--checkpoint-path` but no interval now keeps the provided path and still auto-selects `checkpoint_interval=120.0`.
  - `--no-checkpoint` suppresses both the automatic path and cadence selection.
  - an explicit `--checkpoint-interval` without `--checkpoint-path` now auto-selects the checkpoint directory and reached a real checkpoint-save attempt during the smoke probe.
  - no performance claim is attached to this change; the grounding here is behavioral rather than benchmark-driven.

### March 9, 2026 — `1c67475` — checkpoints: Add checkpoint interval tradeoff tooling — score `4` — complexity `5`

**Human-directed, AI-shaped (4)**

- Requested a checkpoint-frequency tradeoff plot with expected resume-needed frequency on one axis and an optimal target derived from measured checkpoint overhead.
  - Later requested that the practical rule be generalized from discrete hourly/dayly-style thresholds into a logical interval scan anchored at `hourly <= 0.1%` save-only overhead.

**Grounding**

- Files:
  - `.gitignore`
  - `autoresearch_mlx/constants.py`
  - `autoresearch_mlx/checkpoint_policy.py`
  - `tools/checkpoint_tradeoff.py`
- Validation:
  - `python3 -m py_compile tools/checkpoint_tradeoff.py`
  - `env MPLCONFIGDIR=/Users/ent/Codex/autoresearch-everywhere/.mplconfig .venv/bin/python tools/checkpoint_tradeoff.py`
- Measurements:
  - Generated:

    - `results/analysis/checkpoint_tradeoff.png`
    - `results/analysis/checkpoint_tradeoff.csv`
    - `results/analysis/checkpoint_tradeoff.md`
    - `results/analysis/checkpoint_tradeoff.json`
  - Scenario table:

    | Profile                                        | Robustness                            | Events/day | Mean hours between resumes | Optimal interval (min) | Expected waste (%) | Save cost (ms) |
    | ---------------------------------------------- | ------------------------------------- | ---------: | -------------------------: | ---------------------: | -----------------: | -------------: |
    | Exact full-state resume (m5-large calibrated)  | exact step-boundary full-state resume |       0.25 |                      96.00 |                   3.66 |              0.064 |             70 |
    | Exact full-state resume (m5-large calibrated)  | exact step-boundary full-state resume |       1.00 |                      24.00 |                   1.83 |              0.127 |             70 |
    | Exact full-state resume (m5-large calibrated)  | exact step-boundary full-state resume |       2.00 |                      12.00 |                   1.29 |              0.180 |             70 |
    | Exact full-state resume (m5-large calibrated)  | exact step-boundary full-state resume |       4.00 |                       6.00 |                   0.92 |              0.255 |             70 |
    | Exact full-state resume (m5-large calibrated)  | exact step-boundary full-state resume |       8.00 |                       3.00 |                   0.65 |              0.360 |             70 |
    | Exact full-state resume (m5-large calibrated)  | exact step-boundary full-state resume |      24.00 |                       1.00 |                   0.37 |              0.624 |             70 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume |       0.25 |                      96.00 |                   4.59 |              0.080 |            110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume |       1.00 |                      24.00 |                   2.30 |              0.160 |            110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume |       2.00 |                      12.00 |                   1.62 |              0.226 |            110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume |       4.00 |                       6.00 |                   1.15 |              0.319 |            110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume |       8.00 |                       3.00 |                   0.81 |              0.451 |            110 |
    | Exact full-state resume (m5-xlarge calibrated) | exact step-boundary full-state resume |      24.00 |                       1.00 |                   0.47 |              0.782 |            110 |
  - Human-factors interval scan:

    | Profile                                        | Interval | Save-only overhead (%) | Allowed overhead (%) | Pass  |
    | ---------------------------------------------- | -------: | ---------------------: | -------------------: | ----- |
    | Exact full-state resume (m5-large calibrated)  |       1m |                 0.1167 |                0.100 | False |
    | Exact full-state resume (m5-large calibrated)  |       2m |                 0.0583 |                0.100 | True  |
    | Exact full-state resume (m5-xlarge calibrated) |       1m |                 0.1833 |                0.100 | False |
    | Exact full-state resume (m5-xlarge calibrated) |       2m |                 0.0917 |                0.100 | True  |
  - Human-factors recommendations:

    | Profile                                        | Recommended max interval | Reason                                                                                                                           |
    | ---------------------------------------------- | -----------------------: | -------------------------------------------------------------------------------------------------------------------------------- |
    | Exact full-state resume (m5-large calibrated)  |                       2m | 2m is the shortest friendly interval under the fixed 0.100% save-only overhead cap (1m fail); hourly anchor overhead is 0.0019%. |
    | Exact full-state resume (m5-xlarge calibrated) |                       2m | 2m is the shortest friendly interval under the fixed 0.100% save-only overhead cap (1m fail); hourly anchor overhead is 0.0031%. |
  - Tradeoff image: [results/analysis/checkpoint_tradeoff.png](results/analysis/checkpoint_tradeoff.png)
  - Using the currently measured exact full-state resume costs:

    - `m5-large` calibration (`70 ms/save`): optimal interval is about `1.83 min` at `1` resume/day and `0.92 min` at `4` resumes/day
    - `m5-xlarge` calibration (`110 ms/save`): optimal interval is about `2.30 min` at `1` resume/day and `1.15 min` at `4` resumes/day
    - under the generalized human-factors scan anchored at `0.1%` save-only overhead, both grounded profiles recommend a practical maximum checkpoint interval of `2 minutes`
  - The current `2s` benchmark interval is intentionally much more aggressive than the modeled optimum for realistic interruption rates; it remains useful for stress-testing checkpoint overhead, not as the recommended steady-state policy.
  - The tool is structured for multiple robustness profiles, but today only the exact step-boundary full-state resume profile is grounded well enough to include by default.

### March 9, 2026 — `a0d765d` — checkpoints: Add MLX checkpoints and benchmark grounding — score `3.67` — complexity `11`

**Human-driven (5)**

- Requested that, given the constrained Apple Silicon hardware, the canonical matched benchmark run length for grounding optimization changes be `60s` instead of `30s`.

**Human-directed, AI-shaped (4)**

- Requested stronger grounding on the checkpoint/resume work and asked that changelog/program guidance become more explicit about preferring high-signal validation, while leaving the exact comparison design and write-up to the agent.

**AI-identified within brief, human-approved (2)**

- Requested that work continue to the next optimization item after landing the lazy-cache checkpoint.
  - Added resumable checkpoint save/load for the MLX trainer, including model weights, optimizer state, runtime counters, and train-loader cursor state.
  - Added periodic step-boundary checkpoint saves plus explicit `--resume-from`, `--checkpoint-path`, and `--checkpoint-interval` flags.
  - Added train-loader serialization for both prepacked and live-packed paths so resume continues from the saved training position instead of restarting the data stream.

**Grounding**

- Files:
  - `autoresearch_mlx/checkpoints.py`
  - `autoresearch_mlx/data.py`
  - `autoresearch_mlx/train.py`
  - `README.md`
  - `docs/mlx-port-architecture.md`
  - `docs/program-mlx.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/data.py autoresearch_mlx/checkpoints.py`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke --checkpoint-path /tmp/autoresearch_resume_smoke2 --checkpoint-interval 0.5`
  - `./.venv/bin/python -m autoresearch_mlx.train --resume-from /tmp/autoresearch_resume_smoke2 --time-budget 1.5`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke --checkpoint-path /tmp/autoresearch_resume_smoke3 --checkpoint-interval 0.5`
  - matched `60s` preset reruns with `--eval-tokens 512 --canonical-eval-tokens 512` for:
    - `m5-fast`
    - `m5-balanced`
    - `m5-large`
    - `m5-xlarge`
- Measurements:
  - matched `m5-large` run (`20s`, `512` eval tokens) with and without periodic checkpoint saves every `2s`:
    - wall-clock overhead outside tracked training time: `0.1s -> 0.8s` (`+0.7s`) across ten checkpoint writes, or about `0.07s/save`
    - peak memory: `1944.3 MB -> 1944.3 MB` (flat)
    - steady-state per-step throughput stayed in the same `~15k tok/s` band
    - `num_steps` stayed flat at `76 -> 76`, which is the cleaner fixed-budget result
  - matched `m5-xlarge` run (`60s`, `512` eval tokens) with and without periodic checkpoint saves every `2s`:
    - wall-clock overhead outside tracked training time: `0.2s -> 3.5s` (`+3.3s`) across twenty-nine checkpoint writes, or about `0.11s/save`
    - peak memory: `4294.2 MB -> 4294.2 MB` (flat)
    - `num_steps` stayed flat at `115 -> 115`
    - steady-state per-step throughput stayed in the same `~7.8k-8.0k tok/s` band
  - `m5-fast`: `val_bpb=1.964834`, `proxy_val_bpb=1.900012`, `~70.1k tok/s`, `174.5 MB`, `8213` steps
  - `m5-balanced`: `val_bpb=1.757194`, `proxy_val_bpb=1.741310`, `~34.5k tok/s`, `949.9 MB`, `1011` steps
  - `m5-large`: `val_bpb=1.901965`, `proxy_val_bpb=1.935591`, `~14.6k tok/s`, `1944.3 MB`, `215` steps
  - `m5-xlarge`: `val_bpb=2.150093`, `proxy_val_bpb=2.122628`, `~7.8k tok/s`, `4294.2 MB`, `114` steps
- Confirmed behavior:
  - checkpoint saves succeeded during smoke runs without breaking evaluation
  - resume restored the saved state at `step=133`, continued training to `step=186`, and preserved the same train-loader position rather than restarting from zero
  - `m5-fast` and `m5-balanced` used the train-side prepacked cache path during the 60-second reruns
  - `m5-large` and `m5-xlarge` used the token-cache plus live-packing path during those reruns, so their throughput figures are conservative relative to a fully prepacked `1024/2048` cache setup

### March 9, 2026 — `f1d14e8` — model: Lazy-grow model caches — score `2`

**AI-identified within brief, human-approved (2)**

- Requested the next optimization pass on lazy-growing the model caches after the current priority review.
  - Replaced the fixed `sequence_len * 10` RoPE cache with a smaller startup cache that grows up to the configured model sequence length.
  - Reworked local-attention mask caching to keep one growable mask per window size and slice it for shorter requests instead of caching separate masks per exact sequence length.
  - Prewarmed runtime caches in `autoresearch_mlx/train.py` before compiling the train step so cache growth stays out of the compiled hot path.

**Grounding**

- Files:
  - `autoresearch_mlx/model.py`
  - `autoresearch_mlx/train.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py autoresearch_mlx/model.py`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke`
  - `./.venv/bin/python -m autoresearch_mlx.train --preset m5-xlarge --time-budget 0.01 --eval-tokens 512 --canonical-eval-tokens 512`
  - matched A/B benchmark on `m5-xlarge` against the last committed cache behavior from `2be14fe`
- Measurements:
  - matched `m5-xlarge` run (`5s`, `512` eval tokens):
    - completed updates in budget: `10 -> 11`
    - fixed-budget throughput: `7.88k tok/s -> 8.34k tok/s` (`+5.9%`)
    - step-0 latency: `583 ms -> 582 ms` (flat)
    - peak memory: `4298.7 MB -> 4294.2 MB` (effectively flat)
  - ultra-short startup probe (`0.01s`) was noisy and favored the baseline in total wall-clock time, so the longer fixed-budget run is the more reliable comparison
- Confirmed behavior:
  - the smoke path still trains and evaluates end-to-end
  - the `m5-xlarge` path completed with the lazily prewarmed `2048`-token cache setup instead of relying on the old eager `10x` RoPE allocation strategy

### March 9, 2026 — `2be14fe` — changelog: Add changelog score parser — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested a changelog parser for future autonomy plotting and later tightened the requirement to include explicit commit-count tracking for daily per-commit views.
  - Added `tools/changelog_scores.py`.
  - Added per-entry and per-day output.
  - Added CSV/JSON/TSV formats, score verification, and daily `commit_count` plus per-commit score fields.

**Grounding**

- Files:
  - `tools/changelog_scores.py`
  - `CHANGELOG.md`
  - `README.md`
  - `docs/program-mlx.md`
- Validation:
  - `python3 -m py_compile tools/changelog_scores.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify`
  - `python3 tools/changelog_scores.py --group-by day --format csv --include-latest`
- Historical score corrections surfaced by the parser:
  - `9a4ef79`: `39 -> 40`
  - `e06f85c`: `35 -> 36`

### March 9, 2026 — `2bcdc0c` — data: Add optional prepacked row caches — score `3` — complexity `6`

**Human-directed, AI-shaped (4)**

- Implemented split-and-sequence-length keyed prepacked row caches, an opt-in `autoresearch_mlx/prepare.py` build path, runtime preference with fallback, and a trainer flag to disable the caches for ablations.

**AI-identified within brief, human-approved (2)**

- Requested work on optional prepacked caches after landing the optimizer checkpoint.

**Grounding**

- Files:
  - `autoresearch_mlx/constants.py`
  - `autoresearch_mlx/data.py`
  - `autoresearch_mlx/prepare.py`
  - `autoresearch_mlx/train.py`
  - `README.md`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/prepare.py autoresearch_mlx/train.py autoresearch_mlx/data.py autoresearch_mlx/constants.py`
  - `./.venv/bin/python -m autoresearch_mlx.prepare --num-shards 1 --build-prepacked-cache --prepacked-seq-lens 256,512`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke`
  - matched A/B benchmark on `m5-fast` against the live token-cache packing path using `--no-prepacked-cache`
- Measured effect on `m5-fast` (`2s`, matched settings):
  - step-0 latency: `48 ms -> 37 ms` (`-22.9%`)
  - completed updates in budget: `233 -> 255`
  - fixed-budget throughput: `59.65k tok/s -> 65.28k tok/s` (`+9.4%`)
  - peak memory: `147.1 MB -> 147.1 MB` (flat)
  - validation metrics: effectively unchanged within short-run noise
  - confirmed runtime behavior: train and val loaders both switched to `prepacked cache` when the matching cache existed

### March 9, 2026 — `137ba69` — optim: Remove optimizer tree churn — score `3` — complexity `6`

**AI-identified within brief, human-shaped (3)**

- Surfaced optimizer tree churn as the next high-value optimization after streamed gradient accumulation and, after human selection, removed it from the hot path.
  - Reworked `MuonAdamW` to cache stable parameter slots on the model.
  - Fetched gradients by cached path tokens and wrote updated arrays back directly instead of flattening and unflattening the full parameter tree on every step.
  - Cached per-group slot lists so the Muon path no longer rebuilds them inside the hot loop.

**Grounding**

- Files:
  - `autoresearch_mlx/optim.py`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/optim.py`
  - matched A/B benchmark on `m5-large` against the last committed optimizer from `9a4ef79`
- Measured effect on `m5-large` (`5s`, matched settings):
  - step-0 latency: `558 ms -> 365 ms` (`-34.6%`)
  - completed updates in budget: `17 -> 18`
  - fixed-budget throughput: `13.65k tok/s -> 14.18k tok/s` (`+3.8%`)
  - peak memory: `1946.6 MB -> 1946.6 MB` (flat)
  - steady-state per-step throughput: roughly flat within run-to-run noise

### March 9, 2026 — `9a4ef79` — changelog: Add changelog and provenance policy — score `4.25` — complexity `20`

**Human-driven (5)**

- Requested a grounded `CHANGELOG.md` rather than a lightweight release log, with explicit autonomy distinctions and progressively finer provenance tiers.
- Corrected provenance overclaims with a deliberate under-claiming bias and required a branch policy that fully human-authored code changes happen in a fork.

**Human-directed, AI-shaped (4)**

- Added `CHANGELOG.md` and integrated it into the repo workflow.
  - Seeded it with grounded history for the MLX port, evaluation split, preset/token-cache work, and the streamed-accumulation change.
  - Linked the changelog from `README.md`.
  - Updated `docs/program-mlx.md` so future experiment loops keep the changelog current.

**AI-identified within brief, human-shaped (3)**

- Proposed intermediate provenance ladders and wording variants; the human materially reshaped them into the current six-tier taxonomy and policy wording.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `docs/program-mlx.md`
- Validation:
  - docs/process only; no code-path tests were needed

### March 9, 2026 — `077a187` — train: Stream gradient accumulation in MLX trainer — score `3` — complexity `5`

**AI-identified within brief, human-shaped (3)**

- Surfaced streamed gradient accumulation as a likely next optimization target and, after human selection, replaced stacked microbatch gathering with a streamed training loop.
  - Split the train step into a compiled per-microbatch gradient pass plus a compiled gradient-application pass.
  - Updated the architecture report to reflect the new training flow.

**Grounding**

- Files:
  - `autoresearch_mlx/train.py`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py`
  - matched A/B benchmark on `m5-large` against the previous committed training loop from `e06f85c`
- Measured effect on `m5-large` (`5s`, matched settings):
  - steady-state throughput: `15,278.9 tok/s -> 15,279.2 tok/s` (`+0.00%`, effectively flat)
  - peak memory: `2453.8 MB -> 1946.6 MB` (`-507.2 MB`, `-20.7%`)
  - step-0 latency: `533 ms -> 354 ms` (`-33.6%`)
  - steps completed in budget: `18 -> 19`
  - canonical `val_bpb`: `2.373981 -> 2.368727`

### March 9, 2026 — `e06f85c` — data: Add token caching and calibrate M5 presets — score `3.8` — complexity `19`

**Human-driven (5)**

- Requested M5-oriented practical defaults rather than retaining H100-shaped settings.
- Requested measured investigation and documentation of the reference-machine presets, including upstream-scale behavior and the README baseline figures.

**Human-directed, AI-shaped (4)**

- Requested an `m5-xlarge` or "upstream-ish" preset for the tested M5 while leaving the exact shape to the agent, then used fixed-budget measurements to document the concrete throughput, memory, and baseline figures.

**AI-identified within brief, human-shaped (3)**

- Surfaced the time-budget bug during benchmarking and, after the human requested that specific fix, implemented the budget-stop correction for slow presets.

**AI-identified within brief, human-approved (2)**

- Added prepare-time token caches and cache-aware loading so training no longer has to re-tokenize parquet text on every run.

**Grounding**

- Files:
  - `autoresearch_mlx/prepare.py`
  - `autoresearch_mlx/train.py`
  - `autoresearch_mlx/data.py`
  - `README.md`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/prepare.py autoresearch_mlx/train.py tools/overnight_mlx.py autoresearch_mlx/*.py`
  - `./.venv/bin/python -m autoresearch_mlx.prepare --num-shards 1`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke`
  - multiple fixed-budget runs on the reference M5 MacBook Pro (`32 GB`, `10 GPU cores`)
- Measured effect of token caching, isolated by swapping only the data path back to the pre-cache implementation:
  - `m5-fast`:
    - throughput: `61.6k tok/s -> 64.2k tok/s` (`+4.2%`)
    - step-0 latency: `130 ms -> 55 ms` (`-57.7%`)
    - peak memory: `179.6 MB -> 147.1 MB` (`-18.1%`)
  - `m5-balanced`:
    - throughput: `35.2k tok/s -> 36.0k tok/s` (`+2.3%`)
    - step-0 latency: `180 ms -> 132 ms` (`-26.7%`)
    - peak memory: `951.0 MB -> 951.0 MB` (flat in this probe)
- Baseline 5-minute canonical `val_bpb` figures on the reference machine:
  - `m5-fast`: `1.929023`
  - `m5-balanced`: `1.575952`
  - `m5-large`: `1.601354`
  - `m5-xlarge`: `1.765520`
- Upstream-scale ablation on the same machine:
  - literal `upstream` preset ran at roughly `1.1k-1.9k tok/s`
  - the same `50.3M` / `2048` architecture with M5-sized batch (`m5-xlarge`) ran at roughly `7.4k-8.0k tok/s`

### March 9, 2026 — `eee26f5` — train: Separate canonical eval and demote local tooling — score `3` — complexity `9`

**Human-directed, AI-shaped (4)**

- Requested that the fork stay centered on the core MLX research loop and approved demoting optional workstation automation so it would not define the repo.

**AI-identified within brief, human-shaped (3)**

- Surfaced tooling demotion as part of keeping the core MLX path front and center, then moved the overnight automation under `tools/` and rewrote docs to describe it as optional local tooling rather than core architecture.

**AI-identified within brief, human-approved (2)**

- Surfaced canonical-vs-proxy evaluation separation as part of the MLX optimization plan, then split evaluation into canonical `val_bpb` and preset-shaped `proxy_val_bpb` and updated the sweep runner to keep or discard runs using canonical `val_bpb`.

**Grounding**

- Files:
  - `autoresearch_mlx/train.py`
  - `tools/overnight_mlx.py`
  - `tools/launch_overnight_mlx.sh`
  - `tools/detach_exec.py`
  - `README.md`
  - `docs/program-mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/train.py overnight_mlx.py autoresearch_mlx/*.py`
  - `python3 -m py_compile tools/overnight_mlx.py tools/detach_exec.py`
  - `./tools/launch_overnight_mlx.sh tooling-dry-run 0.01 --dry-run`
  - detached one-experiment sweep via the launcher path
- Measured effect:
  - detached sweep produced distinct metrics on the same run:
    - canonical `val_bpb`: `2.509401`
    - proxy `val_bpb`: `2.473907`
  - the sweep runner kept or discarded runs based on canonical `val_bpb`, not the proxy metric

### March 9, 2026 — `c3b3d8d` — mlx: Add initial MLX port for Apple Silicon — score `4` — complexity `7`

**Human-directed, AI-shaped (4)**

- Requested a clean, feature-complete, idiomatic, maintainable MLX reimplementation of upstream `karpathy/autoresearch` for Apple Silicon rather than the PyTorch/MPS SDPA path, then implemented the core MLX path and its Apple-Silicon-first docs.
  - Implemented the MLX data path, model, optimizer, and evaluation path.
  - Kept the upstream CUDA path in-tree for reference while making the MLX path the primary workflow.
  - Added an MLX-specific program file and smoke-test path.

**Grounding**

- Files:
  - `autoresearch_mlx/prepare.py`
  - `autoresearch_mlx/train.py`
  - `autoresearch_mlx/data.py`
  - `autoresearch_mlx/model.py`
  - `autoresearch_mlx/optim.py`
  - `docs/program-mlx.md`
  - `README.md`
  - `pyproject.toml`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/prepare.py autoresearch_mlx/train.py autoresearch_mlx/*.py`
  - `./.venv/bin/python -m autoresearch_mlx.prepare --num-shards 1`
  - `./.venv/bin/python -m autoresearch_mlx.train --smoke`
- Smoke-test result immediately before the baseline commit:
  - canonical `val_bpb`: `2.187049`
  - `training_seconds`: `1.0`
  - `peak_vram_mb`: `194.8`

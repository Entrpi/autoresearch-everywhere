from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

from autoresearch_cuda.checkpoints import load_checkpoint_metadata
from autoresearch_cuda.config import CUDA_PRESETS, resolve_run_preset
from autoresearch_cuda.eval_policy import current_eval_semantics_signature, current_runtime_shape_signature
from autoresearch_cuda.runtime import detect_cuda_runtime_profile, query_nvidia_driver_version
from autoresearch_platform.summary import parse_summary

from .engines import EngineCapabilities, EnginePreset, HardwareFingerprint, ProbeResult


CUDA_FULL_EVAL_TOKENS = 40 * 524288
CUDA_SHORT_PROBE_FWDBWD_TOKENS = 4_096
CUDA_SHORT_PROBE_TOTAL_TOKENS = 8_192
CUDA_SHORT_PROBE_EVAL_TOKENS = 65_536

CUDA_EVAL_RUNG_SPECS = {
    "cheap": {"eval_tokens": 262144, "seq_len": 2048},
    "reference": {"eval_tokens": 3 * 524288, "seq_len": 2048},
    "full": {"eval_tokens": CUDA_FULL_EVAL_TOKENS, "seq_len": 2048},
}


class CUDAEngine:
    name = "cuda"
    backend_family = "cuda"
    reference_preset = "upstream"
    capabilities = EngineCapabilities(
        supports_platform_bringup=True,
        supports_local_search=True,
        supports_eval_calibration=True,
        supports_checkpoint_mint=True,
        supports_runtime_eval_policy=True,
        mutable_axes=("seq_len", "window_pattern", "device_batch_size", "total_batch_size", "depth"),
    )

    def preset_catalog(self) -> dict[str, EnginePreset]:
        return {
            key: EnginePreset(
                key=key,
                description=value.description,
                seq_len=value.seq_len,
                depth=value.depth,
                window_pattern=value.window_pattern,
                device_batch_size=value.device_batch_size,
                total_batch_size=value.total_batch_size,
                tags=("reference",) if key == self.reference_preset else ("starter",),
            )
            for key, value in CUDA_PRESETS.items()
        }

    def preset_order(self) -> tuple[str, ...]:
        return tuple(CUDA_PRESETS.keys())

    def default_platform_presets(self) -> tuple[str, ...]:
        return ("m5-tiny", "m5-small", "m5-balanced", "m5-large", "m5-xlarge")

    def detect_hardware_fingerprint(self) -> HardwareFingerprint:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("CUDA engine requires torch to inspect CUDA hardware.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA engine selected but no CUDA device is available.")

        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        runtime = detect_cuda_runtime_profile(capability, device_name=props.name)
        total_memory_bytes = int(props.total_memory)
        memory_gb = total_memory_bytes / (1024**3)
        driver_version = query_nvidia_driver_version()
        hardware_key = f"nvidia-{runtime.reference_family.family_key}-{int(round(memory_gb))}gb"

        return HardwareFingerprint(
            engine=self.name,
            backend_family=self.backend_family,
            hardware_key=hardware_key,
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            python_version=platform.python_version(),
            runtime_version=getattr(torch.version, "cuda", None),
            driver_version=driver_version,
            accelerator_vendor="NVIDIA",
            accelerator_model=props.name,
            accelerator_architecture=runtime.reference_family.family_key,
            accelerator_compute_capability=f"{capability[0]}.{capability[1]}",
            accelerator_cores=getattr(props, "multi_processor_count", None),
            memory_bytes=total_memory_bytes,
            memory_gb=memory_gb,
            attention_backend=runtime.selected_attention_backend,
            flash_attention_generation=runtime.preferred_flash_attention_generation,
        )

    def run_train_probe(
        self,
        *,
        preset: str,
        time_budget: float,
        logs_dir: Path,
        stage: str,
        benchmark_skip_eval: bool,
        checkpoint_path: Path | None = None,
        seq_len: int | None = None,
        window_pattern: str | None = None,
        device_batch_size: int | None = None,
        total_batch_size: int | None = None,
        curve_eval_seconds: tuple[float, ...] | None = None,
        eval_seq_len: int | None = None,
        eval_tokens: int | None = None,
        eval_batch_size: int | None = None,
        no_checkpoint: bool = True,
    ) -> ProbeResult:
        resolved = resolve_run_preset(
            preset,
            time_budget=time_budget,
            seq_len=seq_len,
            window_pattern=window_pattern,
            total_batch_size=total_batch_size,
            depth=None,
            device_batch_size=device_batch_size,
        )
        grad_accum_steps = self._infer_grad_accum(
            resolved.seq_len,
            resolved.device_batch_size,
            resolved.total_batch_size,
        )
        if grad_accum_steps is None:
            raise ValueError(
                f"Invalid total batch {resolved.total_batch_size} for seq_len={resolved.seq_len}, "
                f"device_batch_size={resolved.device_batch_size}"
            )

        explicit_batch_shape = device_batch_size is not None or total_batch_size is not None
        if explicit_batch_shape:
            resolved_device_batch = resolved.device_batch_size
            resolved_total_batch = resolved.total_batch_size
        else:
            resolved_device_batch, resolved_total_batch = self._short_probe_batch_shape(
                seq_len=resolved.seq_len,
                device_batch_size=resolved.device_batch_size,
                total_batch_size=resolved.total_batch_size,
                time_budget=resolved.time_budget,
                benchmark_skip_eval=benchmark_skip_eval,
            )
        grad_accum_steps = self._infer_grad_accum(
            resolved.seq_len,
            resolved_device_batch,
            resolved_total_batch,
        )
        if grad_accum_steps is None:
            raise ValueError(
                f"Invalid short-probe batch {resolved_total_batch} for seq_len={resolved.seq_len}, "
                f"device_batch_size={resolved_device_batch}"
            )

        cmd = [
            sys.executable,
            "train.py",
            "--preset",
            preset,
            "--time-budget",
            str(resolved.time_budget),
            "--seq-len",
            str(resolved.seq_len),
            "--depth",
            str(resolved.depth),
            "--window-pattern",
            resolved.window_pattern,
            "--device-batch-size",
            str(resolved_device_batch),
            "--total-batch-size",
            str(resolved_total_batch),
        ]
        if benchmark_skip_eval:
            cmd.append("--benchmark-skip-eval")
        # Cold torch.compile dominates tiny CUDA probes on Blackwell-class systems.
        # Use eager mode for short calibration/search runs, and reserve compile for
        # longer measurements where steady-state benefits can amortize.
        if resolved.time_budget <= 5.0:
            cmd.append("--no-compile")
            if not benchmark_skip_eval:
                cmd.extend(
                    [
                        "--eval-seq-len",
                        str(resolved.seq_len),
                        "--eval-tokens",
                        str(CUDA_SHORT_PROBE_EVAL_TOKENS),
                        "--eval-batch-size",
                        str(resolved_device_batch),
                    ]
                )
        if checkpoint_path is not None and not no_checkpoint:
            cmd.extend(["--checkpoint-path", str(checkpoint_path)])
        if no_checkpoint:
            cmd.append("--no-checkpoint")

        label = self._command_label(
            [
                stage,
                preset,
                f"seq{resolved.seq_len}",
                f"db{resolved_device_batch}",
                f"tb{resolved_total_batch}",
                resolved.window_pattern,
            ]
        )
        curve_output_path = logs_dir / f"{label}.curve.json" if curve_eval_seconds else None
        if curve_eval_seconds:
            cmd.extend(["--curve-eval-seconds", ",".join(f"{value:g}" for value in curve_eval_seconds)])
            cmd.extend(["--curve-output", str(curve_output_path)])
        if eval_seq_len is not None:
            cmd.extend(["--eval-seq-len", str(eval_seq_len)])
        if eval_tokens is not None:
            cmd.extend(["--eval-tokens", str(eval_tokens)])
        if eval_batch_size is not None:
            cmd.extend(["--eval-batch-size", str(eval_batch_size)])
        completed, wall_seconds, stdout_path, stderr_path = self._run_command(cmd, logs_dir=logs_dir, label=label)
        summary = parse_summary(completed.stdout) if completed.returncode == 0 else {}
        error_tail = None
        if completed.returncode != 0:
            error_tail = "\n".join(completed.stderr.splitlines()[-12:])

        steady_state_tok_per_sec = self._get_float(summary, "steady_state_tok_per_sec")
        if steady_state_tok_per_sec is None:
            training_seconds = self._get_float(summary, "training_seconds")
            total_tokens_m = self._get_float(summary, "total_tokens_M")
            if training_seconds and total_tokens_m is not None and training_seconds > 0:
                steady_state_tok_per_sec = (total_tokens_m * 1_000_000.0) / training_seconds
        peak_flop_utilization_percent = self._get_float(summary, "peak_flop_utilization_percent")
        if peak_flop_utilization_percent is None:
            peak_flop_utilization_percent = self._get_float(summary, "mfu_percent")

        return ProbeResult(
            preset=preset,
            stage=stage,
            seq_len=resolved.seq_len,
            depth=resolved.depth,
            window_pattern=resolved.window_pattern,
            device_batch_size=resolved_device_batch,
            total_batch_size=resolved_total_batch,
            grad_accum_steps=grad_accum_steps,
            status="ok" if completed.returncode == 0 else "error",
            returncode=completed.returncode,
            wall_seconds=wall_seconds,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            curve_output_path=str(curve_output_path) if curve_output_path is not None else self._get_str(summary, "curve_output"),
            curve_eval_points=self._get_int(summary, "curve_eval_points"),
            val_bpb=self._get_float(summary, "val_bpb"),
            steady_state_tok_per_sec=steady_state_tok_per_sec,
            peak_vram_mb=self._get_float(summary, "peak_vram_mb"),
            training_seconds=self._get_float(summary, "training_seconds"),
            total_seconds=self._get_float(summary, "total_seconds"),
            eval_percent=self._get_float(summary, "eval_percent"),
            input_pipeline_percent=self._get_float(summary, "input_pipeline_percent"),
            optimizer_percent=self._get_float(summary, "optimizer_percent"),
            forward_backward_percent=self._get_float(summary, "forward_backward_percent"),
            other_step_percent=self._get_float(summary, "other_step_percent"),
            compute_share_percent=self._get_float(summary, "compute_share_percent"),
            train_tflops=self._get_float(summary, "train_tflops"),
            peak_flop_utilization_percent=peak_flop_utilization_percent,
            util_window_steps=self._get_int(summary, "util_window_steps"),
            util_window=self._get_str(summary, "util_window"),
            canonical_rung=self._get_str(summary, "canonical_rung"),
            canonical_seq_len=self._get_int(summary, "canonical_eval_seq_len") or self._get_int(summary, "eval_seq_len"),
            canonical_tokens=self._get_int(summary, "canonical_eval_tokens") or self._get_int(summary, "eval_tokens"),
            canonical_batch=self._get_int(summary, "canonical_eval_batch_size") or self._get_int(summary, "eval_batch_size"),
            eval_calibration_status=self._get_str(summary, "eval_calibration_status"),
            eval_calibration_effective_confidence=self._get_str(summary, "eval_calibration_effective_confidence"),
            eval_calibration_freshness=self._get_str(summary, "eval_calibration_freshness"),
            eval_calibration_limited_by=self._get_str(summary, "eval_calibration_limited_by"),
            error_tail=error_tail,
        )

    def default_local_seq_lens(self, preset: str, *, mode: str) -> list[int]:
        preset_config = CUDA_PRESETS[preset]
        if mode == "fast":
            return [preset_config.seq_len]
        candidates = [preset_config.seq_len]
        halved = max(512, preset_config.seq_len // 2)
        if halved != preset_config.seq_len:
            candidates.append(halved)
        return sorted(set(candidates))

    def default_local_window_patterns(self, preset: str, *, mode: str) -> list[str]:
        preset_config = CUDA_PRESETS[preset]
        patterns = [preset_config.window_pattern]
        if mode == "full" and preset_config.window_pattern != "L":
            patterns.append("L")
        return list(dict.fromkeys(patterns))

    def _minimum_search_device_batch(self, *, seq_len: int) -> int:
        return 2 if seq_len >= 2048 else 4

    def local_batch_candidates(self, preset: str, *, seq_len: int) -> list[tuple[int, int]]:
        value = CUDA_PRESETS[preset]
        base_device_batch = value.device_batch_size
        min_device_batch = self._minimum_search_device_batch(seq_len=seq_len)
        device_batches = sorted({
            min_device_batch,
            max(min_device_batch, base_device_batch // 4),
            max(min_device_batch, base_device_batch // 2),
            base_device_batch,
            base_device_batch * 2,
        })
        grad_accum_candidates = (1, 2)
        combos: list[tuple[int, int]] = []
        for device_batch in device_batches:
            tokens_per_fwdbwd = seq_len * device_batch
            if tokens_per_fwdbwd <= 0:
                continue
            for grad_accum in grad_accum_candidates:
                combos.append((device_batch, tokens_per_fwdbwd * grad_accum))
        return sorted(set(combos))

    def batch_profile_candidates(self, preset: str, *, seq_len: int) -> list[tuple[int, int]]:
        value = CUDA_PRESETS[preset]
        base_device_batch = value.device_batch_size
        min_device_batch = self._minimum_search_device_batch(seq_len=seq_len)
        device_batches = sorted(
            {
                min_device_batch,
                max(min_device_batch, base_device_batch // 4),
                max(min_device_batch, base_device_batch // 2),
                base_device_batch,
                base_device_batch * 2,
            }
        )
        grad_accum_candidates = (1, 2, 4, 8)
        combos: list[tuple[int, int]] = []
        for device_batch in device_batches:
            tokens_per_fwdbwd = seq_len * device_batch
            if tokens_per_fwdbwd <= 0:
                continue
            for grad_accum in grad_accum_candidates:
                combos.append((device_batch, tokens_per_fwdbwd * grad_accum))
        return sorted(set(combos))

    def batch_profile_refinement_candidates(
        self,
        preset: str,
        *,
        seq_len: int,
        coarse_winner: tuple[int, int],
    ) -> list[tuple[int, int]]:
        winner_device_batch, winner_total_batch = coarse_winner
        tokens_per_fwdbwd = seq_len * winner_device_batch
        if tokens_per_fwdbwd <= 0:
            return []
        winner_grad_accum = max(1, winner_total_batch // tokens_per_fwdbwd)
        min_device_batch = self._minimum_search_device_batch(seq_len=seq_len)
        device_batches = {
            min_device_batch,
            max(min_device_batch, winner_device_batch // 2),
            winner_device_batch,
            winner_device_batch * 2,
        }
        grad_accum_candidates = {
            max(1, winner_grad_accum - 2),
            max(1, winner_grad_accum - 1),
            winner_grad_accum,
            winner_grad_accum + 1,
            winner_grad_accum + 2,
            max(1, round(winner_grad_accum * 1.5)),
            winner_grad_accum * 2,
        }
        combos: list[tuple[int, int]] = []
        for device_batch in sorted(device_batches):
            tokens_per_fwdbwd = seq_len * device_batch
            if tokens_per_fwdbwd <= 0:
                continue
            for grad_accum in sorted(acc for acc in grad_accum_candidates if acc <= 16):
                combos.append((device_batch, tokens_per_fwdbwd * grad_accum))
        return sorted(set(combos))

    def calibration_signatures(self) -> dict[str, str | None]:
        return {
            "eval_semantics_signature": current_eval_semantics_signature(),
            "runtime_shape_signature": current_runtime_shape_signature(),
        }

    def run_eval_calibration(
        self,
        *,
        preset: str,
        checkpoint_dir: Path,
        hardware_key: str,
        rungs: list[str],
        budget_seconds: list[float],
        markdown_path: Path | None = None,
    ) -> dict:
        resolved = resolve_run_preset(preset)
        checkpoint_metadata = load_checkpoint_metadata(checkpoint_dir)
        checkpoint_run_config = checkpoint_metadata.get("run_config", {})
        checkpoint_device_batch_size = int(
            checkpoint_run_config.get("device_batch_size", resolved.device_batch_size)
        )
        rows: list[dict] = []

        for rung_key in rungs:
            try:
                spec = CUDA_EVAL_RUNG_SPECS[rung_key]
            except KeyError as exc:
                raise ValueError(f"Unknown CUDA eval rung: {rung_key}") from exc

            eval_batch_size = self._default_eval_batch_size(checkpoint_device_batch_size)

            cmd = [
                sys.executable,
                "train.py",
                "--preset",
                preset,
                "--resume-from",
                str(checkpoint_dir),
                "--eval-only",
                "--eval-seq-len",
                str(spec["seq_len"]),
                "--eval-tokens",
                str(spec["eval_tokens"]),
                "--eval-batch-size",
                str(eval_batch_size),
                "--no-compile",
                "--no-checkpoint",
            ]
            label = self._command_label(
                [
                    "eval-rung",
                    preset,
                    rung_key,
                    f"seq{spec['seq_len']}",
                    f"eb{eval_batch_size}",
                    f"tok{spec['eval_tokens']}",
                ]
            )
            logs_dir = checkpoint_dir / "eval_calibration_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            completed, wall_seconds, stdout_path, stderr_path = self._run_command(cmd, logs_dir=logs_dir, label=label)
            if completed.returncode != 0:
                error_tail = "\n".join(completed.stderr.splitlines()[-12:])
                raise RuntimeError(
                    f"CUDA eval calibration rung {rung_key!r} failed for preset {preset!r}.\n"
                    f"stdout: {stdout_path}\nstderr: {stderr_path}\n{error_tail}"
                )
            summary = parse_summary(completed.stdout)
            val_bpb = self._get_float(summary, "val_bpb")
            eval_seconds = self._get_float(summary, "eval_seconds")
            seq_len = self._get_int(summary, "eval_seq_len")
            eval_tokens = self._get_int(summary, "eval_tokens")
            batch_size = self._get_int(summary, "eval_batch_size")
            if val_bpb is None or eval_seconds is None or seq_len is None or batch_size is None:
                raise RuntimeError(
                    f"CUDA eval calibration rung {rung_key!r} did not emit the required summary fields.\n"
                    f"stdout: {stdout_path}\nstderr: {stderr_path}"
                )
            rows.append(
                {
                    "rung": rung_key,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "eval_tokens": eval_tokens if eval_tokens is not None else spec["eval_tokens"],
                    "val_bpb": val_bpb,
                    "eval_seconds": eval_seconds,
                    "wall_seconds": wall_seconds,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )

        full_row = next((row for row in rows if row["rung"] == "full"), None)
        if full_row is not None:
            full_bpb = float(full_row["val_bpb"])
            full_seconds = float(full_row["eval_seconds"])
            for row in rows:
                row["abs_error_vs_full"] = abs(float(row["val_bpb"]) - full_bpb)
                row["speedup_vs_full"] = (
                    full_seconds / float(row["eval_seconds"])
                    if float(row["eval_seconds"]) > 0
                    else None
                )
                for budget in budget_seconds:
                    row[f"overhead_{int(budget)}s"] = float(row["eval_seconds"]) / budget

        payload = {
            "mode": "eval-rungs",
            "preset": preset,
            "hardware_key": hardware_key,
            "checkpoint": str(checkpoint_dir),
            "eval_semantics_signature": current_eval_semantics_signature(),
            "runtime_shape_signature": current_runtime_shape_signature(),
            "rows": rows,
        }
        if markdown_path is not None:
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(self._render_eval_calibration_markdown(payload))
        return payload

    @staticmethod
    def _command_label(parts: list[str]) -> str:
        return "_".join(
            part.replace("--", "").replace("/", "_").replace("=", "_").replace(",", "_")
            for part in parts
        )

    @staticmethod
    def _run_command(cmd: list[str], *, logs_dir: Path, label: str):
        stdout_path = logs_dir / f"{label}.stdout.log"
        stderr_path = logs_dir / f"{label}.stderr.log"
        started = time.perf_counter()
        completed = subprocess.run(cmd, capture_output=True, text=True)
        wall_seconds = time.perf_counter() - started
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        return completed, wall_seconds, stdout_path, stderr_path

    @staticmethod
    def _infer_grad_accum(seq_len: int, device_batch_size: int, total_batch_size: int) -> int | None:
        tokens_per_fwdbwd = seq_len * device_batch_size
        if tokens_per_fwdbwd <= 0 or total_batch_size % tokens_per_fwdbwd != 0:
            return None
        return total_batch_size // tokens_per_fwdbwd

    @staticmethod
    def _short_probe_batch_shape(
        *,
        seq_len: int,
        device_batch_size: int,
        total_batch_size: int,
        time_budget: float,
        benchmark_skip_eval: bool,
    ) -> tuple[int, int]:
        if time_budget > 5.0:
            return device_batch_size, total_batch_size

        target_device_batch = max(1, CUDA_SHORT_PROBE_FWDBWD_TOKENS // seq_len)
        probe_device_batch = min(device_batch_size, target_device_batch)
        tokens_per_fwdbwd = seq_len * probe_device_batch
        target_total_tokens = CUDA_SHORT_PROBE_TOTAL_TOKENS
        probe_total_batch = min(total_batch_size, max(tokens_per_fwdbwd, target_total_tokens))
        probe_total_batch = max(tokens_per_fwdbwd, (probe_total_batch // tokens_per_fwdbwd) * tokens_per_fwdbwd)
        return probe_device_batch, probe_total_batch

    @staticmethod
    def _get_float(summary: dict, key: str) -> float | None:
        value = summary.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_int(summary: dict, key: str) -> int | None:
        value = summary.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_str(summary: dict, key: str) -> str | None:
        value = summary.get(key)
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _default_eval_batch_size(device_batch_size: int) -> int:
        return max(1, min(32, device_batch_size))

    @staticmethod
    def _render_eval_calibration_markdown(payload: dict) -> str:
        rows = payload["rows"]
        lines = [
            "# CUDA Eval Calibration",
            "",
            f"- Preset: `{payload['preset']}`",
            f"- Hardware: `{payload['hardware_key']}`",
            "",
            "| Rung | Seq len | Batch | Eval tokens | Val BPB | Eval seconds | Abs error vs full |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            abs_error = row.get("abs_error_vs_full")
            abs_error_text = f"{abs_error:.6f}" if abs_error is not None else "n/a"
            lines.append(
                f"| `{row['rung']}` | {row['seq_len']} | {row['batch_size']} | "
                f"{row['eval_tokens']} | {row['val_bpb']:.6f} | {row['eval_seconds']:.2f} | {abs_error_text} |"
            )
        return "\n".join(lines) + "\n"

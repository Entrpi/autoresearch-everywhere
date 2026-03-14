from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CudaReferenceFamily:
    family_key: str
    family_name: str
    status: str
    notes: str | None = None


@dataclass(frozen=True)
class CudaArchitectureProfile:
    architecture_key: str
    architecture_name: str
    compute_capability: tuple[int, int]
    preferred_flash_attention_generation: int | None
    selected_flash_attention_repo: str
    selected_attention_backend: str
    reference_family: CudaReferenceFamily


CURRENT_CUDA_REFERENCE_FAMILIES: tuple[CudaReferenceFamily, ...] = (
    CudaReferenceFamily("ampere-sm80", "Ampere SM80 (A100/A800)", "current"),
    CudaReferenceFamily(
        "ada-rtx40",
        "Ada RTX 40xx-class",
        "current",
        notes="Consumer Ada is kept separate from L40S-class Ada because workstation/datacenter envelopes differ materially.",
    ),
    CudaReferenceFamily("ada-l40s", "Ada L40S-class", "current"),
    CudaReferenceFamily("hopper-sm90", "Hopper SM90 (H100/H200)", "current"),
    CudaReferenceFamily(
        "blackwell-rtx50",
        "Blackwell RTX 50xx-class",
        "current",
        notes="Kept separate from datacenter Blackwell because consumer/workstation operating envelopes differ materially.",
    ),
    CudaReferenceFamily("blackwell-b200", "Blackwell B200-class", "current"),
    CudaReferenceFamily(
        "blackwell-gb10",
        "Blackwell GB10 (DGX Spark-class)",
        "current",
        notes="Kept separate from B200-class Blackwell because its system envelope and likely calibration posture differ.",
    ),
)

ANTICIPATED_CUDA_REFERENCE_FAMILIES: tuple[CudaReferenceFamily, ...] = (
    CudaReferenceFamily(
        "rubin-unknown",
        "Vera Rubin (anticipated)",
        "anticipated",
        notes="Tracked as a future CUDA reference family without hardcoding compute capability or FA runtime policy yet.",
    ),
)


def classify_cuda_architecture(major: int, minor: int) -> tuple[str, str]:
    if (major, minor) == (8, 0):
        return ("ampere-sm80", "Ampere SM80")
    if major >= 10:
        return ("blackwell-sm100", "Blackwell SM100+")
    if (major, minor) == (9, 0):
        return ("hopper-sm90", "Hopper SM90")
    if (major, minor) == (8, 9):
        return ("ada-sm89", "Ada SM89 family")
    if major == 8:
        return ("ampere", "Ampere")
    if (major, minor) == (7, 5):
        return ("turing", "Turing")
    if major == 7:
        return ("volta", "Volta")
    if major == 6:
        return ("pascal", "Pascal")
    return (f"sm_{major}{minor}", f"SM {major}.{minor}")


def preferred_flash_attention_generation(capability: tuple[int, int]) -> int | None:
    major, minor = capability
    if (major, minor) == (8, 0):
        return 2
    if (major, minor) == (8, 9):
        return 2
    if major >= 10:
        return 4
    if major >= 9:
        return 3
    return None


def select_flash_attention_repo(
    capability: tuple[int, int], *, device_name: str | None = None
) -> tuple[str, str]:
    if capability == (8, 0):
        return ("Dao-AILab/flash-attention", "flash-attn2")
    if capability == (8, 9):
        return ("Dao-AILab/flash-attention", "flash-attn2")
    if capability == (9, 0):
        return ("varunneal/flash-attention-3", "flash-attn3")
    if capability[0] >= 10:
        normalized = _normalized_device_name(device_name)
        if "gb10" in normalized or "dgx spark" in normalized:
            return ("Dao-AILab/flash-attention#2268", "flash-attn4")
        return ("Dao-AILab/flash-attention", "flash-attn4")
    return ("kernels-community/flash-attn3", "flash-attn3")


def _normalized_device_name(device_name: str | None) -> str:
    if device_name is None:
        return ""
    return " ".join(device_name.lower().replace("-", " ").split())


def reference_family_for_capability(capability: tuple[int, int], *, device_name: str | None = None) -> CudaReferenceFamily:
    major, minor = capability
    if (major, minor) == (8, 0):
        return CURRENT_CUDA_REFERENCE_FAMILIES[0]
    if (major, minor) == (8, 9):
        normalized = _normalized_device_name(device_name)
        if "l40s" in normalized:
            return CURRENT_CUDA_REFERENCE_FAMILIES[2]
        return CURRENT_CUDA_REFERENCE_FAMILIES[1]
    if (major, minor) == (9, 0):
        return CURRENT_CUDA_REFERENCE_FAMILIES[3]
    if major >= 10:
        normalized = _normalized_device_name(device_name)
        if "rtx 50" in normalized or "geforce rtx 5" in normalized:
            return CURRENT_CUDA_REFERENCE_FAMILIES[4]
        if "gb10" in normalized or "dgx spark" in normalized:
            return CURRENT_CUDA_REFERENCE_FAMILIES[6]
        return CURRENT_CUDA_REFERENCE_FAMILIES[5]
    architecture_key, architecture_name = classify_cuda_architecture(major, minor)
    return CudaReferenceFamily(architecture_key, architecture_name, "current")


def detect_cuda_runtime_profile(capability: tuple[int, int], *, device_name: str | None = None) -> CudaArchitectureProfile:
    architecture_key, architecture_name = classify_cuda_architecture(*capability)
    repo, backend = select_flash_attention_repo(capability, device_name=device_name)
    return CudaArchitectureProfile(
        architecture_key=architecture_key,
        architecture_name=architecture_name,
        compute_capability=capability,
        preferred_flash_attention_generation=preferred_flash_attention_generation(capability),
        selected_flash_attention_repo=repo,
        selected_attention_backend=backend,
        reference_family=reference_family_for_capability(capability, device_name=device_name),
    )


def query_nvidia_driver_version() -> str | None:
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.splitlines()[0]
            .strip()
        )
    except Exception:
        return None

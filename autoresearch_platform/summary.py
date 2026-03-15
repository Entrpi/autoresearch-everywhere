from __future__ import annotations


def parse_summary(stdout: str) -> dict[str, str | float | int]:
    result: dict[str, str | float | int] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        value = raw.strip()
        if not value:
            continue
        try:
            if any(char in value for char in ".eE"):
                parsed: str | float | int = float(value)
            else:
                parsed = int(value)
        except ValueError:
            parsed = value
        result[key] = parsed
    return result


def parse_final_summary(stdout: str) -> dict[str, str | float | int]:
    normalized = stdout.replace("\r", "\n")
    marker = "\n---\n"
    marker_index = normalized.rfind(marker)
    if marker_index < 0:
        raise ValueError("Could not find final summary marker '---' in training output.")
    summary_block = normalized[marker_index + len(marker) :]
    return parse_summary(summary_block)

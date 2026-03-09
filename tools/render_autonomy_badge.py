#!/usr/bin/env python3
"""
Render the README autonomy golf badge and snapshot from the current changelog rollup.

Example:
    python3 tools/render_autonomy_badge.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "autonomy-golf-badge.svg"
README_PATH = ROOT / "README.md"

LABEL_TEXT = "autonomy golf"
LABEL_WIDTH = 108
BADGE_HEIGHT = 20
MAX_SCORE = 6
SNAPSHOT_START = "<!-- autonomy-golf-snapshot:start -->"
SNAPSHOT_END = "<!-- autonomy-golf-snapshot:end -->"
HOUSE_TERMS = {
    0: "hole in one",
    1: "albatross",
    2: "eagle",
    3: "birdie",
    4: "par",
    5: "bogey",
    6: "double bogey",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the autonomy golf README badge.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the SVG badge.",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Do not update the README snapshot block.",
    )
    return parser.parse_args()


def badge_color(score: float) -> str:
    if score <= 1.0:
        return "#2ea44f"
    if score <= 2.0:
        return "#3fb950"
    if score <= 3.0:
        return "#9a6700"
    if score <= 4.0:
        return "#d29922"
    if score <= 5.0:
        return "#db6d28"
    return "#cf222e"


def estimate_width(text: str) -> int:
    return max(46, int(len(text) * 6.8) + 16)


def house_term(score: float) -> str:
    nearest = min(MAX_SCORE, max(0, int(score + 0.5)))
    return HOUSE_TERMS[nearest]


def read_overall_metrics() -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "changelog_scores.py"),
            "--group-by",
            "overall",
            "--format",
            "json",
            "--include-latest",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    rows = json.loads(result.stdout)
    if not rows:
        raise RuntimeError("No autonomy golf rows were produced.")
    return rows[0]


def render_svg(*, score_text: str, color: str, title: str) -> str:
    value_width = estimate_width(score_text)
    total_width = LABEL_WIDTH + value_width
    label_x = LABEL_WIDTH / 2 + 8
    value_x = LABEL_WIDTH + value_width / 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{BADGE_HEIGHT}" role="img" aria-label="{title}">
  <title>{title}</title>
  <linearGradient id="badge-fill" x2="0" y2="100%">
    <stop offset="0" stop-color="#ffffff" stop-opacity=".08"/>
    <stop offset="1" stop-opacity=".08"/>
  </linearGradient>
  <mask id="badge-mask">
    <rect width="{total_width}" height="{BADGE_HEIGHT}" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#badge-mask)">
    <rect width="{LABEL_WIDTH}" height="{BADGE_HEIGHT}" fill="#24292f"/>
    <rect x="{LABEL_WIDTH}" width="{value_width}" height="{BADGE_HEIGHT}" fill="{color}"/>
    <rect width="{total_width}" height="{BADGE_HEIGHT}" fill="url(#badge-fill)"/>
  </g>
  <g fill="none" stroke="#fff" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="10" y1="5" x2="10" y2="15"/>
    <path d="M10 5 L17 7.5 L10 10 Z" fill="#fff" stroke="none"/>
    <path d="M7.5 15.5 Q10 13.8 12.5 15.5" opacity="0.85"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif" font-size="11">
    <text x="{label_x}" y="15">autonomy golf</text>
    <text x="{value_x}" y="15">{score_text}</text>
  </g>
</svg>
"""


def render_snapshot_markdown(row: dict[str, object]) -> str:
    commits = int(row["commit_count"])
    subsystems = int(row["subsystem_count"])
    commit_label = "commit" if commits == 1 else "commits"
    subsystem_label = "subsystem" if subsystems == 1 else "subsystems"
    return "\n".join(
        [
            SNAPSHOT_START,
            "Current project snapshot from [CHANGELOG.md](CHANGELOG.md):",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Mean autonomy score | `{float(row['mean_computed_score']):.2f} / {MAX_SCORE}` |",
            f"| Mean complexity | `{float(row['mean_computed_complexity']):.2f} / commit` |",
            f"| Mean score per top-level bullet | `{float(row['mean_score_per_bullet']):.2f} / {MAX_SCORE}` |",
            f"| History covered | `{commits}` {commit_label} across `{subsystems}` {subsystem_label} |",
            SNAPSHOT_END,
        ]
    )


def update_readme_snapshot(row: dict[str, object]) -> None:
    if not README_PATH.exists():
        raise RuntimeError(f"README not found at {README_PATH}")
    readme = README_PATH.read_text()
    start = readme.find(SNAPSHOT_START)
    end = readme.find(SNAPSHOT_END)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("README snapshot markers not found or malformed.")
    end += len(SNAPSHOT_END)
    block = render_snapshot_markdown(row)
    README_PATH.write_text(readme[:start] + block + readme[end:])


def main() -> int:
    args = parse_args()
    row = read_overall_metrics()
    score = float(row["mean_computed_score"])
    complexity = float(row["mean_computed_complexity"])
    commits = int(row["commit_count"])
    term = house_term(score)
    score_text = f"{term} {score:.2f}/{MAX_SCORE}"
    title = (
        f"Autonomy golf: {term}, {score:.2f}/{MAX_SCORE} mean score, "
        f"{complexity:.2f} mean complexity across {commits} commits"
    )
    svg = render_svg(score_text=score_text, color=badge_color(score), title=title)
    args.output.write_text(svg)
    if not args.skip_readme:
        update_readme_snapshot(row)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

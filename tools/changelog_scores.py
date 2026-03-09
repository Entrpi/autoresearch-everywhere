#!/usr/bin/env python3
"""
Parse the autonomy-golf changelog and emit rollups.

Examples:
    python3 tools/changelog_scores.py --group-by overall --format csv --include-latest
    python3 tools/changelog_scores.py --group-by subsystem --format json --include-latest
    python3 tools/changelog_scores.py --group-by entry --format csv --include-latest --verify
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SCORE_PRECISION = 2
ROOT = Path(__file__).resolve().parents[1]
LATEST_SECTION_NAME = "Latest"

CATEGORY_ORDER = (
    "Fully human",
    "Human-driven",
    "Human-directed, AI-shaped",
    "AI-identified within brief, human-shaped",
    "AI-identified within brief, human-approved",
    "Self-initiated, human-approved",
    "Fully autonomous",
)

CATEGORY_WEIGHTS = {
    "Fully human": 6,
    "Human-driven": 5,
    "Human-directed, AI-shaped": 4,
    "AI-identified within brief, human-shaped": 3,
    "AI-identified within brief, human-approved": 2,
    "Self-initiated, human-approved": 1,
    "Fully autonomous": 0,
}

CATEGORY_HEADING_RE = re.compile(r"^(?P<label>.+?)(?: \((?P<score>\d+)\))?$")
SUBSYSTEM_TITLE_RE = re.compile(r"^(?P<subsystem>[a-z0-9][a-z0-9_./-]*): (?P<summary>.+)$")
NARRATIVE_PREFIXES = ("Meaning:", "Motivation:", "Purpose:")

EM_DASH = "\u2014"
HEADER_RE = re.compile(
    rf"^### (?:(?P<date_label>.+?) {EM_DASH} )?"
    rf"(?:(?P<commit>`[^`]+`) {EM_DASH} )?"
    rf"(?P<title>.+?)"
    rf"(?: {EM_DASH} score `(?P<score>[^`]+)`)"
    rf"(?: {EM_DASH} complexity `(?P<complexity>[^`]+)`)?$"
)


@dataclass
class Entry:
    section: str
    title: str
    header_score: float
    header_complexity: int | None
    commit: str | None
    subsystem: str
    summary: str
    is_latest: bool
    date_value: date | None
    date_label: str | None
    current_section: str | None = None
    category_counts: dict[str, int] | None = None
    nested_bonus_count: int = 0

    def __post_init__(self) -> None:
        if self.category_counts is None:
            self.category_counts = {category: 0 for category in CATEGORY_ORDER}

    @property
    def top_level_bullet_count(self) -> int:
        return sum(self.category_counts.values())

    @property
    def total_points(self) -> int:
        return sum(self.category_counts[category] * CATEGORY_WEIGHTS[category] for category in CATEGORY_ORDER)

    @property
    def computed_complexity(self) -> int:
        return self.total_points + self.nested_bonus_count

    @property
    def computed_score(self) -> float:
        if self.top_level_bullet_count == 0:
            return 0.0
        return round_score(self.total_points / self.top_level_bullet_count)

    @property
    def score_delta(self) -> float:
        return round_score(self.header_score - self.computed_score)

    @property
    def complexity_delta(self) -> int:
        return self.effective_header_complexity - self.computed_complexity

    @property
    def has_explicit_complexity(self) -> bool:
        return self.header_complexity is not None

    @property
    def effective_header_complexity(self) -> int:
        if self.header_complexity is not None:
            return self.header_complexity
        if self.header_score.is_integer():
            return int(self.header_score)
        return int(round(self.header_score))

    def as_row(self) -> dict[str, object]:
        row = {
            "section": self.section,
            "date": self.date_value.isoformat() if self.date_value else "",
            "date_label": self.date_label or "",
            "commit": self.commit or "",
            "subsystem": self.subsystem,
            "title": self.title,
            "summary": self.summary,
            "header_score": self.header_score,
            "computed_score": self.computed_score,
            "score_delta": self.score_delta,
            "header_complexity": self.effective_header_complexity,
            "computed_complexity": self.computed_complexity,
            "complexity_delta": self.complexity_delta,
            "has_explicit_complexity": self.has_explicit_complexity,
            "is_latest": self.is_latest,
            "top_level_bullet_count": self.top_level_bullet_count,
            "total_points": self.total_points,
            "nested_bonus_count": self.nested_bonus_count,
        }
        for category in CATEGORY_ORDER:
            slug = slugify(category)
            count = self.category_counts[category]
            row[f"{slug}_count"] = count
            row[f"{slug}_points"] = count * CATEGORY_WEIGHTS[category]
        return row


def round_score(value: float) -> float:
    return round(value, SCORE_PRECISION)


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def parse_header(line: str, section: str) -> Entry:
    match = HEADER_RE.match(line)
    if match is None:
        raise ValueError(f"Could not parse changelog header: {line}")
    raw_score = match.group("score")
    if raw_score is None:
        raise ValueError(f"Header is missing score: {line}")
    title = match.group("title")
    title_match = SUBSYSTEM_TITLE_RE.match(title)
    if title_match is None:
        raise ValueError(f"Header is missing subsystem prefix: {line}")
    date_label = match.group("date_label")
    commit = match.group("commit")
    date_value = None
    if date_label and date_label != "New commit":
        date_value = date.fromisoformat(parse_date_label(date_label))
    return Entry(
        section=section,
        title=title,
        header_score=float(raw_score),
        header_complexity=int(match.group("complexity")) if match.group("complexity") else None,
        commit=commit.strip("`") if commit else None,
        subsystem=title_match.group("subsystem"),
        summary=title_match.group("summary"),
        is_latest=section == LATEST_SECTION_NAME,
        date_value=date_value,
        date_label=date_label,
    )


def parse_date_label(date_label: str) -> str:
    month_name, day_str, year_str = date_label.replace(",", "").split()
    month = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }[month_name]
    day = int(day_str)
    year = int(year_str)
    return f"{year:04d}-{month:02d}-{day:02d}"


def is_section_heading(line: str) -> bool:
    return line.startswith("## ")


def is_entry_heading(line: str) -> bool:
    return line.startswith("### ")


def parse_category_heading(line: str) -> str | None:
    if not line.startswith("**") or not line.endswith("**"):
        return None
    label = line.strip("*")
    match = CATEGORY_HEADING_RE.match(label)
    if match is None:
        return None
    label = match.group("label")
    if label not in CATEGORY_WEIGHTS and label != "Grounding":
        return None
    return label


def is_narrative_nested_bullet(line: str) -> bool:
    stripped = line[4:].strip()
    return any(stripped.startswith(prefix) for prefix in NARRATIVE_PREFIXES)


def parse_changelog(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    current_top_level: str | None = None
    current_entry: Entry | None = None
    section: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        if is_section_heading(line):
            section = line[3:]
            continue
        if is_entry_heading(line):
            if current_entry is not None:
                entries.append(current_entry)
            if section is None:
                raise ValueError(f"Entry found before section heading: {line}")
            current_entry = parse_header(line, section)
            current_top_level = None
            continue
        if current_entry is None:
            continue
        category = parse_category_heading(line)
        if category is not None:
            current_entry.current_section = category
            current_top_level = category if category in CATEGORY_WEIGHTS else None
            continue
        if line.startswith("- ") and current_entry.current_section in CATEGORY_WEIGHTS:
            current_entry.category_counts[current_entry.current_section] += 1
            continue
        if (
            line.startswith("  - ")
            and current_entry.current_section in CATEGORY_WEIGHTS
            and CATEGORY_WEIGHTS[current_entry.current_section] >= 3
            and not is_narrative_nested_bullet(line)
        ):
            current_entry.nested_bonus_count += 1

    if current_entry is not None:
        entries.append(current_entry)
    return entries


def group_by_date(entries: list[Entry]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for entry in entries:
        key = entry.date_value.isoformat() if entry.date_value else "latest"
        if key not in grouped:
            grouped[key] = {
                "date": key,
                "commit_count": 0,
                "top_level_bullet_count": 0,
                "total_points": 0,
                "total_nested_bonus": 0,
                "total_complexity": 0,
                "total_header_score": 0,
                "total_computed_score": 0,
                "total_score_delta": 0,
                "total_header_complexity": 0,
                "total_complexity_delta": 0,
                **{f"{slugify(category)}_count": 0 for category in CATEGORY_ORDER},
                **{f"{slugify(category)}_points": 0 for category in CATEGORY_ORDER},
            }
        row = grouped[key]
        row["commit_count"] += 1
        row["top_level_bullet_count"] += entry.top_level_bullet_count
        row["total_points"] += entry.total_points
        row["total_nested_bonus"] += entry.nested_bonus_count
        row["total_complexity"] += entry.computed_complexity
        row["total_header_score"] += entry.header_score
        row["total_computed_score"] += entry.computed_score
        row["total_score_delta"] += entry.score_delta
        row["total_header_complexity"] += entry.effective_header_complexity
        row["total_complexity_delta"] += entry.complexity_delta
        for category in CATEGORY_ORDER:
            slug = slugify(category)
            row[f"{slug}_count"] += entry.category_counts[category]
            row[f"{slug}_points"] += entry.category_counts[category] * CATEGORY_WEIGHTS[category]

    rows = []
    for row in grouped.values():
        commit_count = row["commit_count"]
        row["mean_header_score"] = round_score(row["total_header_score"] / commit_count)
        row["mean_computed_score"] = round_score(row["total_computed_score"] / commit_count)
        row["mean_score_per_bullet"] = round_score(row["total_points"] / row["top_level_bullet_count"])
        row["mean_header_complexity"] = round_score(row["total_header_complexity"] / commit_count)
        row["mean_computed_complexity"] = round_score(row["total_complexity"] / commit_count)
        rows.append(row)
    return sorted(rows, key=lambda row: row["date"])


def init_grouped_score_row() -> dict[str, object]:
    row = {
        "commit_count": 0,
        "top_level_bullet_count": 0,
        "total_points": 0,
        "total_nested_bonus": 0,
        "total_complexity": 0,
        "total_header_score": 0,
        "total_computed_score": 0,
        "total_score_delta": 0,
        "total_header_complexity": 0,
        "total_complexity_delta": 0,
    }
    for category in CATEGORY_ORDER:
        slug = slugify(category)
        row[f"{slug}_count"] = 0
        row[f"{slug}_points"] = 0
    return row


def accumulate_grouped_score_row(row: dict[str, object], entry: Entry) -> None:
    row["commit_count"] += 1
    row["top_level_bullet_count"] += entry.top_level_bullet_count
    row["total_points"] += entry.total_points
    row["total_nested_bonus"] += entry.nested_bonus_count
    row["total_complexity"] += entry.computed_complexity
    row["total_header_score"] += entry.header_score
    row["total_computed_score"] += entry.computed_score
    row["total_score_delta"] += entry.score_delta
    row["total_header_complexity"] += entry.effective_header_complexity
    row["total_complexity_delta"] += entry.complexity_delta
    for category in CATEGORY_ORDER:
        slug = slugify(category)
        row[f"{slug}_count"] += entry.category_counts[category]
        row[f"{slug}_points"] += entry.category_counts[category] * CATEGORY_WEIGHTS[category]


def finalize_grouped_score_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in rows:
        commit_count = row["commit_count"]
        row["mean_header_score"] = round_score(row["total_header_score"] / commit_count)
        row["mean_computed_score"] = round_score(row["total_computed_score"] / commit_count)
        row["mean_score_per_bullet"] = round_score(row["total_points"] / row["top_level_bullet_count"])
        row["mean_header_complexity"] = round_score(row["total_header_complexity"] / commit_count)
        row["mean_computed_complexity"] = round_score(row["total_complexity"] / commit_count)
    return rows


def group_by_subsystem(entries: list[Entry]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for entry in entries:
        row = grouped.setdefault(entry.subsystem, {"subsystem": entry.subsystem, **init_grouped_score_row()})
        accumulate_grouped_score_row(row, entry)
    rows = list(grouped.values())
    rows.sort(key=lambda row: row["subsystem"])
    return finalize_grouped_score_rows(rows)


def group_by_day_subsystem(entries: list[Entry]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        day = entry.date_value.isoformat() if entry.date_value else "latest"
        key = (day, entry.subsystem)
        row = grouped.setdefault(
            key,
            {"date": day, "subsystem": entry.subsystem, **init_grouped_score_row()},
        )
        accumulate_grouped_score_row(row, entry)
    rows = list(grouped.values())
    rows.sort(key=lambda row: (row["date"], row["subsystem"]))
    return finalize_grouped_score_rows(rows)


def group_overall(entries: list[Entry]) -> list[dict[str, object]]:
    row = init_grouped_score_row()
    subsystems = set()
    for entry in entries:
        accumulate_grouped_score_row(row, entry)
        subsystems.add(entry.subsystem)
    row["scope"] = "including_latest" if any(entry.is_latest for entry in entries) else "committed_only"
    row["subsystem_count"] = len(subsystems)
    finalize_grouped_score_rows([row])
    return [row]


def emit_csv(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def emit_json(rows: list[dict[str, object]]) -> None:
    json.dump(rows, sys.stdout, indent=2)
    sys.stdout.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and summarize the autonomy-golf changelog.")
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "CHANGELOG.md",
        help="Path to the changelog file.",
    )
    parser.add_argument(
        "--group-by",
        choices=("entry", "day", "subsystem", "day-subsystem", "overall"),
        default="entry",
        help="Rollup dimension.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--include-latest",
        action="store_true",
        help="Include entries from the Latest section.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fail if header scores or subsystem prefixes do not match computed values.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = parse_changelog(args.path)
    if not args.include_latest:
        entries = [entry for entry in entries if not entry.is_latest]

    if args.verify:
        mismatches = [entry for entry in entries if abs(entry.score_delta) > 1e-9 or entry.complexity_delta != 0]
        unscoped = [entry for entry in entries if not SUBSYSTEM_TITLE_RE.match(entry.title)]
        if mismatches or unscoped:
            for entry in mismatches:
                commit = entry.commit or "latest"
                print(
                    f"score mismatch: {commit} {entry.title!r} "
                    f"header={entry.header_score} computed={entry.computed_score}",
                    file=sys.stderr,
                )
                print(
                    f"complexity mismatch: {commit} {entry.title!r} "
                    f"header={entry.effective_header_complexity} computed={entry.computed_complexity}",
                    file=sys.stderr,
                )
            if unscoped:
                print("unscoped entries:", file=sys.stderr)
                for entry in unscoped:
                    print(f"  {entry.commit or 'latest'} {entry.title}", file=sys.stderr)
            return 1

    if args.group_by == "entry":
        rows = [entry.as_row() for entry in entries]
    elif args.group_by == "day":
        rows = group_by_date(entries)
    elif args.group_by == "subsystem":
        rows = group_by_subsystem(entries)
    elif args.group_by == "day-subsystem":
        rows = group_by_day_subsystem(entries)
    else:
        rows = group_overall(entries)

    if args.format == "csv":
        emit_csv(rows)
    else:
        emit_json(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

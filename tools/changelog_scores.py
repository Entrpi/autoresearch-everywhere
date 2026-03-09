#!/usr/bin/env python3
"""
Parse CHANGELOG.md autonomy scores into plotting-friendly rows.

Examples:
    python3 tools/changelog_scores.py --group-by day --format csv
    python3 tools/changelog_scores.py --group-by subsystem --format csv
    python3 tools/changelog_scores.py --group-by day-subsystem --format csv
    python3 tools/changelog_scores.py --group-by entry --format json --include-unreleased --verify
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


CATEGORY_ORDER = (
    "Human-driven",
    "Human-directed, AI-shaped",
    "AI-identified within brief, human-shaped",
    "AI-identified within brief, human-approved",
    "Self-initiated, human-approved",
    "Fully autonomous",
)

CATEGORY_WEIGHTS = {
    "Human-driven": 5,
    "Human-directed, AI-shaped": 4,
    "AI-identified within brief, human-shaped": 3,
    "AI-identified within brief, human-approved": 2,
    "Self-initiated, human-approved": 1,
    "Fully autonomous": 0,
}

CATEGORY_HEADING_RE = re.compile(r"^(?P<label>.+?)(?: \((?P<score>\d+)\))?$")
SUBSYSTEM_TITLE_RE = re.compile(r"^(?P<subsystem>[a-z0-9][a-z0-9_./-]*): (?P<summary>.+)$")

EM_DASH = "\u2014"
HEADER_RE = re.compile(rf"^### (?P<prefix>.+?) {EM_DASH} score `(?P<score>\d+)`$")
COMMITTED_PREFIX_RE = re.compile(
    rf"^(?P<date>[A-Za-z]+ \d{{1,2}}, \d{{4}}) {EM_DASH} `(?P<commit>[0-9a-f]+)` {EM_DASH} (?P<title>.+)$"
)


@dataclass
class Entry:
    section: str
    raw_header: str
    title: str
    subsystem: str | None
    summary: str
    header_score: int
    commit: str | None
    date_label: str | None
    current_section: str | None = None
    category_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.category_counts is None:
            self.category_counts = {category: 0 for category in CATEGORY_ORDER}

    @property
    def computed_score(self) -> int:
        return sum(self.category_counts[category] * CATEGORY_WEIGHTS[category] for category in CATEGORY_ORDER)

    @property
    def score_delta(self) -> int:
        return self.header_score - self.computed_score

    @property
    def is_unreleased(self) -> bool:
        return self.section == "Unreleased"

    @property
    def missing_subsystem_prefix(self) -> bool:
        return self.subsystem is None

    @property
    def iso_date(self) -> str | None:
        if self.date_label is None:
            return None
        return datetime.strptime(self.date_label, "%B %d, %Y").date().isoformat()

    def as_row(self) -> dict[str, object]:
        row = {
            "section": self.section,
            "date": self.iso_date,
            "date_label": self.date_label,
            "commit": self.commit,
            "subsystem": self.subsystem,
            "title": self.title,
            "summary": self.summary,
            "header_score": self.header_score,
            "computed_score": self.computed_score,
            "score_delta": self.score_delta,
            "is_unreleased": self.is_unreleased,
        }
        for category in CATEGORY_ORDER:
            slug = slugify(category)
            count = self.category_counts[category]
            row[f"{slug}_count"] = count
            row[f"{slug}_points"] = count * CATEGORY_WEIGHTS[category]
        return row


def slugify(label: str) -> str:
    return label.lower().replace(",", "").replace("-", "").replace(" ", "_")


def normalize_section_label(label: str) -> str:
    match = CATEGORY_HEADING_RE.match(label)
    if not match:
        return label
    base_label = match.group("label")
    score_text = match.group("score")
    if score_text is not None and base_label in CATEGORY_WEIGHTS:
        expected = CATEGORY_WEIGHTS[base_label]
        seen = int(score_text)
        if seen != expected:
            raise ValueError(
                f"Section heading score mismatch for {base_label!r}: saw {seen}, expected {expected}"
            )
    return base_label


def parse_entry_header(line: str, current_section: str) -> Entry:
    match = HEADER_RE.match(line)
    if not match:
        raise ValueError(f"Unrecognized changelog header: {line}")

    prefix = match.group("prefix")
    header_score = int(match.group("score"))
    committed = COMMITTED_PREFIX_RE.match(prefix)
    if committed:
        title = committed.group("title")
        commit = committed.group("commit")
        date_label = committed.group("date")
    else:
        title = prefix.split(f" {EM_DASH} ", 1)[1] if f" {EM_DASH} " in prefix else prefix
        commit = None
        date_label = None

    subsystem_match = SUBSYSTEM_TITLE_RE.match(title)
    subsystem = subsystem_match.group("subsystem") if subsystem_match else None
    summary = subsystem_match.group("summary") if subsystem_match else title
    return Entry(
        section=current_section,
        raw_header=line,
        title=title,
        subsystem=subsystem,
        summary=summary,
        header_score=header_score,
        commit=commit,
        date_label=date_label,
    )


def parse_changelog(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    current_top_level: str | None = None
    current_entry: Entry | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current_top_level = line[3:].strip()
            continue
        if line.startswith("### "):
            if current_entry is not None:
                entries.append(current_entry)
            if current_top_level is None:
                raise ValueError(f"Entry header found outside a top-level section: {line}")
            current_entry = parse_entry_header(line, current_top_level)
            continue
        if current_entry is None:
            continue
        if line.startswith("**") and line.endswith("**"):
            label = normalize_section_label(line.strip("*"))
            current_entry.current_section = label
            continue
        if line.startswith("- ") and current_entry.current_section in CATEGORY_WEIGHTS:
            current_entry.category_counts[current_entry.current_section] += 1

    if current_entry is not None:
        entries.append(current_entry)
    return entries


def build_daily_rows(entries: list[Entry], include_unreleased: bool) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for entry in entries:
        key = entry.iso_date
        if key is None:
            if not include_unreleased:
                continue
            key = "unreleased"
        if key not in grouped:
            grouped[key] = {
                "date": key,
                "commit_count": 0,
                "total_header_score": 0,
                "total_computed_score": 0,
                "total_score_delta": 0,
            }
            for category in CATEGORY_ORDER:
                slug = slugify(category)
                grouped[key][f"{slug}_count"] = 0
                grouped[key][f"{slug}_points"] = 0
        row = grouped[key]
        row["commit_count"] += 1
        row["total_header_score"] += entry.header_score
        row["total_computed_score"] += entry.computed_score
        row["total_score_delta"] += entry.score_delta
        for category in CATEGORY_ORDER:
            slug = slugify(category)
            count = entry.category_counts[category]
            row[f"{slug}_count"] += count
            row[f"{slug}_points"] += count * CATEGORY_WEIGHTS[category]
    rows = []
    for key in sorted(grouped):
        row = grouped[key]
        commit_count = row["commit_count"]
        row["header_score_per_commit"] = row["total_header_score"] / commit_count
        row["computed_score_per_commit"] = row["total_computed_score"] / commit_count
        rows.append(row)
    return rows


def init_grouped_score_row(**fields: object) -> dict[str, object]:
    row = {
        "commit_count": 0,
        "total_header_score": 0,
        "total_computed_score": 0,
        "total_score_delta": 0,
        **fields,
    }
    for category in CATEGORY_ORDER:
        slug = slugify(category)
        row[f"{slug}_count"] = 0
        row[f"{slug}_points"] = 0
    return row


def accumulate_grouped_score_row(row: dict[str, object], entry: Entry) -> None:
    row["commit_count"] += 1
    row["total_header_score"] += entry.header_score
    row["total_computed_score"] += entry.computed_score
    row["total_score_delta"] += entry.score_delta
    for category in CATEGORY_ORDER:
        slug = slugify(category)
        count = entry.category_counts[category]
        row[f"{slug}_count"] += count
        row[f"{slug}_points"] += count * CATEGORY_WEIGHTS[category]


def finalize_grouped_rows(grouped: dict[object, dict[str, object]], sort_keys: list[object]) -> list[dict[str, object]]:
    rows = []
    for key in sort_keys:
        row = grouped[key]
        commit_count = row["commit_count"]
        row["header_score_per_commit"] = row["total_header_score"] / commit_count
        row["computed_score_per_commit"] = row["total_computed_score"] / commit_count
        rows.append(row)
    return rows


def build_subsystem_rows(entries: list[Entry], include_unreleased: bool) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for entry in entries:
        if entry.is_unreleased and not include_unreleased:
            continue
        key = entry.subsystem or "unscoped"
        if key not in grouped:
            grouped[key] = init_grouped_score_row(subsystem=key)
        accumulate_grouped_score_row(grouped[key], entry)
    return finalize_grouped_rows(grouped, sorted(grouped))


def build_day_subsystem_rows(entries: list[Entry], include_unreleased: bool) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        date_key = entry.iso_date
        if date_key is None:
            if not include_unreleased:
                continue
            date_key = "unreleased"
        subsystem_key = entry.subsystem or "unscoped"
        key = (date_key, subsystem_key)
        if key not in grouped:
            grouped[key] = init_grouped_score_row(date=date_key, subsystem=subsystem_key)
        accumulate_grouped_score_row(grouped[key], entry)
    return finalize_grouped_rows(grouped, sorted(grouped))


def emit_rows(rows: list[dict[str, object]], fmt: str) -> None:
    if fmt == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    if fmt == "csv":
        writer.writeheader()
        writer.writerows(rows)
        return

    if fmt == "tsv":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        return

    raise ValueError(f"Unsupported format: {fmt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize autonomy scores from CHANGELOG.md.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="Path to the changelog file.",
    )
    parser.add_argument(
        "--group-by",
        choices=("entry", "day", "subsystem", "day-subsystem"),
        default="day",
        help="Emit one row per commit entry, day, subsystem, or day+subsystem.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv", "tsv"),
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--include-unreleased",
        action="store_true",
        help="Include unreleased entries. Day output groups them under 'unreleased'.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Exit nonzero if any header score does not match the computed score.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = parse_changelog(args.path)

    if args.verify:
        mismatches = [entry for entry in entries if entry.score_delta != 0]
        unscoped = [entry for entry in entries if entry.missing_subsystem_prefix]
        if mismatches:
            for entry in mismatches:
                commit = entry.commit or "unreleased"
                print(
                    f"score mismatch: {commit} {entry.title!r} header={entry.header_score} computed={entry.computed_score}",
                    file=sys.stderr,
                )
        if unscoped:
            for entry in unscoped:
                commit = entry.commit or "unreleased"
                print(f"missing subsystem prefix: {commit} {entry.title!r}", file=sys.stderr)
        if mismatches or unscoped:
            return 1

    if not args.include_unreleased:
        entries = [entry for entry in entries if not entry.is_unreleased]

    if args.group_by == "entry":
        rows = [entry.as_row() for entry in entries]
    elif args.group_by == "day":
        rows = build_daily_rows(entries, args.include_unreleased)
    elif args.group_by == "subsystem":
        rows = build_subsystem_rows(entries, args.include_unreleased)
    else:
        rows = build_day_subsystem_rows(entries, args.include_unreleased)
    emit_rows(rows, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

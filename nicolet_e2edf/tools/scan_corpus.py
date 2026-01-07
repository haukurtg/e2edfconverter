"""Scan a corpus of Nicolet files and report event/annotation coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nicolet_e2edf.nicolet.header import read_nervus_header


def _collect_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(root.glob(pattern))
    return [path for path in files if path.is_file()]


def _scan_file(path: Path) -> dict[str, object]:
    try:
        _, header = read_nervus_header(path)
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {
            "file": str(path),
            "format": None,
            "channels": 0,
            "segments": 0,
            "events": 0,
            "annotations": 0,
            "unknown_event_types": 0,
            "error": str(exc),
        }

    events = header.Events or []
    annotations = [event for event in events if event.annotation]
    unknown = [event for event in events if event.IDStr == "UNKNOWN"]
    channel_count = len(header.Segments[0].chName) if header.Segments else 0
    return {
        "file": str(path),
        "format": header.format or "nicolet-e",
        "channels": channel_count,
        "segments": len(header.Segments),
        "events": len(events),
        "annotations": len(annotations),
        "unknown_event_types": len(unknown),
        "error": "",
    }


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    fields = [
        "file",
        "format",
        "channels",
        "segments",
        "events",
        "annotations",
        "unknown_event_types",
        "error",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(rows: list[dict[str, object]], output: Path) -> None:
    payload = {"files": rows}
    output.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan Nicolet .e/.eeg files for event/annotation coverage."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("EEG_test_files/eeg_files"),
        help="Root folder to scan",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.e",
        help="Glob pattern to match files (e.g., '**/*.e' or '**/*.{e,eeg}')",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Write CSV output to this file")
    parser.add_argument("--json", type=Path, default=None, help="Write JSON output to this file")
    args = parser.parse_args()

    root = args.root
    files = _collect_files(root, args.pattern)
    rows = [_scan_file(path) for path in files]

    totals = {
        "files": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "events": sum(int(row["events"]) for row in rows),
        "annotations": sum(int(row["annotations"]) for row in rows),
        "unknown_event_types": sum(int(row["unknown_event_types"]) for row in rows),
    }

    print("files,errors,events,annotations,unknown_event_types")
    print(
        f"{totals['files']},{totals['errors']},{totals['events']},"
        f"{totals['annotations']},{totals['unknown_event_types']}"
    )

    if args.csv:
        _write_csv(rows, args.csv)
    if args.json:
        _write_json(rows, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

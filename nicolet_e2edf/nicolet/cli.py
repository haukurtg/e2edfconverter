from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path

from .data import read_nervus_data
from .edf_writer import write_edf
from .header import read_nervus_header

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nicolet-e2edf",
        description="Convert Nicolet/Nervus .e EEG recordings into EDF files.",
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Input .e file or folder")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--glob",
        default="*.e",
        help="Glob pattern when input is a folder (default also picks up .eeg files)",
    )
    parser.add_argument(
        "--patient-json",
        type=Path,
        help="Optional JSON file with patient metadata glob rules",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


def _default_patient_metadata(path: Path) -> dict[str, str]:
    digest = hashlib.sha1(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()[:8].upper()
    return {
        "PatientName": "Anon",
        "PatientID": f"SUBJ-{digest}",
        "PatientSex": "O",
        "StudyDescription": "Nicolet EEG export",
        "SeriesDescription": "Nicolet EEG export",
    }


def _load_patient_rules(path: Path | None) -> list[dict[str, str]]:
    if not path:
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - handled in CLI execution
        raise SystemExit(f"Unable to parse patient JSON: {exc}") from exc
    if isinstance(data, dict):
        data = [data]
    rules: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict) or "glob" not in entry:
            raise SystemExit("Each patient rule must be an object with a 'glob' key")
        rules.append(entry)
    return rules


def _select_patient_metadata(path: Path, rules: Iterable[dict[str, str]]) -> dict[str, str]:
    base = _default_patient_metadata(path)
    for rule in rules:
        pattern = rule.get("glob")
        if not isinstance(pattern, str):
            continue
        if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
            override = {k: v for k, v in rule.items() if k != "glob"}
            base.update(override)
            break
    return base


def _discover_inputs(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        matches = sorted(input_path.glob(glob_pattern))
        if glob_pattern == "*.e":
            # Default discovery also includes Nicolet exports saved with a .eeg extension.
            matches = sorted({*matches, *input_path.glob("*.eeg")})
        return matches
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _channel_labels(header, channels: Iterable[int]) -> list[str]:
    labels = header.Segments[0].chName if header.Segments else []
    resolved: list[str] = []
    for channel in channels:
        zero_based = channel - 1
        try:
            label = labels[zero_based]
        except (IndexError, TypeError):
            label = f"Ch{channel}"
        cleaned = label.split("\x00", 1)[0].strip()
        resolved.append(cleaned or f"Ch{channel}")
    return resolved


def _select_channels(header) -> list[int]:
    if header.matchingChannels:
        return list(header.matchingChannels)
    return list(range(1, len(header.TSInfo) + 1))


def convert_file(
    input_path: Path,
    output_dir: Path,
    patient_rules: list[dict[str, str]],
) -> Path:
    public_header, nrv_header = read_nervus_header(input_path)
    fs = public_header.get("Fs") or nrv_header.targetSamplingRate
    if not fs:
        raise RuntimeError("Unable to determine sampling frequency from header")

    channels = _select_channels(nrv_header)
    if not channels:
        raise RuntimeError("No channels available for conversion")

    channel_labels = _channel_labels(nrv_header, channels)
    waveform = read_nervus_data(input_path, nrv_header, channels=channels).T  # samples x channels
    patient_meta = _select_patient_metadata(input_path, patient_rules)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.edf"
    write_edf(
        output_path,
        waveform,
        fs,
        channel_labels,
        patient_meta,
        recording_start=nrv_header.startDateTime,
    )
    logger.info(
        "Converted %s → %s (%d channels @ %.2f Hz)",
        input_path.name,
        output_path.name,
        len(channel_labels),
        fs,
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    input_path = Path(args.input_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    patient_rules = _load_patient_rules(args.patient_json)

    try:
        inputs = _discover_inputs(input_path, args.glob)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not inputs:
        logger.error("No input files matched the supplied path/pattern")
        return 1

    success = 0
    for path in inputs:
        try:
            convert_file(path, output_dir, patient_rules)
            success += 1
        except Exception as exc:  # pragma: no cover - logged for CLI feedback
            logger.error("Failed to convert %s: %s", path, exc)
    if success == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

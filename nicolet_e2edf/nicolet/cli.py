from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .data import read_nervus_data
from .edf_writer import write_edf
from .header import read_nervus_header
from .tui import TuiOptions, rich_available, run_rich_wizard, run_tui

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nicolet-e2edf",
        description="Convert Nicolet/Nervus .e EEG recordings into EDF files.",
    )
    parser.add_argument(
        "--in",
        dest="input_paths",
        nargs="*",
        help="Input .e file(s) and/or folder(s)",
    )
    parser.add_argument("--out", dest="output_dir", help="Output directory")
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
    parser.add_argument(
        "--json-sidecar",
        action="store_true",
        help="Write a JSON sidecar with channel metadata and events",
    )
    parser.add_argument(
        "--resample-to",
        type=float,
        help="Optional sampling rate (Hz) to resample the output EDF to",
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=None,
        help="Optional high-pass filter cutoff in Hz (e.g. 0.5)",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=None,
        help="Optional low-pass filter cutoff in Hz (e.g. 35)",
    )
    parser.add_argument(
        "--notch",
        type=float,
        default=None,
        help="Optional notch filter frequency in Hz (e.g. 50 or 60 for powerline)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--ui",
        dest="ui",
        action="store_true",
        help="Interactive terminal UI (wizard + animated progress; requires 'rich')",
    )
    # Backwards-compatible aliases (hidden): use --ui instead.
    parser.add_argument("--gui", dest="ui", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wiz", dest="ui", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tui", dest="ui", action="store_true", help=argparse.SUPPRESS)
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
        candidates: set[Path] = set()

        def _add(pattern: str) -> None:
            candidates.update(input_path.glob(pattern))

        normalized = glob_pattern.lower()
        if normalized.endswith(".e"):
            _add(glob_pattern)
            _add(glob_pattern[:-2] + ".eeg")
        elif normalized.endswith(".eeg"):
            _add(glob_pattern)
            _add(glob_pattern[:-4] + ".e")
        else:
            # Treat the glob as a name/path filter, and always target .e/.eeg.
            _add(f"{glob_pattern}.e")
            _add(f"{glob_pattern}.eeg")

        return sorted(candidates)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _discover_inputs_with_roots(
    input_paths: Iterable[Path],
    glob_pattern: str,
) -> list[tuple[Path, Path | None]]:
    discovered: list[tuple[Path, Path | None]] = []
    for input_path in input_paths:
        if input_path.is_file():
            discovered.append((input_path, None))
            continue
        if input_path.is_dir():
            for match in _discover_inputs(input_path, glob_pattern):
                discovered.append((match, input_path))
            continue
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return discovered


def _candidate_output_path(input_path: Path, output_dir: Path, input_root: Path | None) -> Path:
    if input_root and input_root.is_dir():
        try:
            relative = input_path.relative_to(input_root)
        except ValueError:
            relative = Path(input_path.name)
        return output_dir / relative.with_suffix(".edf")
    return output_dir / f"{input_path.stem}.edf"


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


def _resample_waveform(waveform: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if target_fs <= 0:
        raise ValueError("Resample rate must be positive")
    if np.isclose(original_fs, target_fs):
        return waveform

    n_samples, n_channels = waveform.shape
    duration_seconds = n_samples / float(original_fs)
    target_samples = max(int(round(duration_seconds * target_fs)), 1)
    original_times = np.arange(n_samples, dtype=np.float64) / float(original_fs)
    target_times = np.arange(target_samples, dtype=np.float64) / float(target_fs)

    resampled = np.zeros((target_samples, n_channels), dtype=waveform.dtype)
    for idx in range(n_channels):
        resampled[:, idx] = np.interp(target_times, original_times, waveform[:, idx])
    return resampled


def _bandpass_filter(
    waveform: np.ndarray,
    fs: float,
    lowcut: float | None,
    highcut: float | None,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass/highpass/lowpass filter.
    
    Args:
        waveform: Shape (samples, channels) array of signal data.
        fs: Sampling frequency in Hz.
        lowcut: High-pass cutoff frequency in Hz (or None to skip).
        highcut: Low-pass cutoff frequency in Hz (or None to skip).
    
    Returns:
        Filtered waveform with same shape as input.
    """
    # If no filtering requested, return as-is
    if lowcut is None and highcut is None:
        return waveform
    
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:
        raise RuntimeError(
            "Filtering requires scipy. Install it with: uv pip install -p .venv scipy"
        ) from exc
    
    nyquist = fs / 2.0
    order = 4  # Standard order for EEG filtering
    
    # Determine filter type and normalized frequencies
    if lowcut is not None and highcut is not None:
        # Bandpass filter
        low = lowcut / nyquist
        high = highcut / nyquist
        # Clamp to valid range (0, 1)
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        if low >= high:
            raise ValueError(f"lowcut ({lowcut} Hz) must be less than highcut ({highcut} Hz)")
        b, a = butter(order, [low, high], btype="band")
    elif lowcut is not None:
        # High-pass filter only
        low = lowcut / nyquist
        low = max(0.001, min(low, 0.999))
        b, a = butter(order, low, btype="high")
    else:
        # Low-pass filter only (highcut is not None)
        high = highcut / nyquist
        high = max(0.001, min(high, 0.999))
        b, a = butter(order, high, btype="low")
    
    # Apply zero-phase filtering to each channel
    n_samples, n_channels = waveform.shape
    filtered = np.zeros_like(waveform)
    for idx in range(n_channels):
        filtered[:, idx] = filtfilt(b, a, waveform[:, idx])
    
    return filtered


def _notch_filter(
    waveform: np.ndarray,
    fs: float,
    notch_freq: float | None,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Apply a zero-phase notch filter to remove powerline interference.
    
    Args:
        waveform: Shape (samples, channels) array of signal data.
        fs: Sampling frequency in Hz.
        notch_freq: Center frequency to notch out (e.g. 50 or 60 Hz), or None to skip.
        quality_factor: Quality factor Q (higher = narrower notch). Default 30.
    
    Returns:
        Filtered waveform with same shape as input.
    """
    if notch_freq is None or notch_freq <= 0:
        return waveform
    
    try:
        from scipy.signal import iirnotch, filtfilt
    except ImportError as exc:
        raise RuntimeError(
            "Filtering requires scipy. Install it with: uv pip install -p .venv scipy"
        ) from exc
    
    # Design notch filter
    w0 = notch_freq / (fs / 2.0)  # Normalized frequency
    if w0 >= 1.0:
        # Notch frequency is at or above Nyquist - skip
        return waveform
    
    b, a = iirnotch(w0, quality_factor)
    
    # Apply zero-phase filtering to each channel
    n_samples, n_channels = waveform.shape
    filtered = np.zeros_like(waveform)
    for idx in range(n_channels):
        filtered[:, idx] = filtfilt(b, a, waveform[:, idx])
    
    return filtered


def _write_json_sidecar(
    output_path: Path,
    *,
    source_path: Path,
    sampling_rate: float,
    channel_labels: list[str],
    sample_count: int,
    start_time,
    events,
) -> None:
    start_iso = start_time.isoformat() if start_time else None
    event_payload = []
    if events:
        for event in events:
            onset = None
            if start_time and event.date:
                onset = (event.date - start_time).total_seconds()
            event_payload.append(
                {
                    "label": event.label,
                    "annotation": event.annotation,
                    "user": event.user,
                    "guid": event.GUID,
                    "id": event.IDStr,
                    "onset_seconds": onset,
                    "duration_seconds": event.duration if event.duration else None,
                }
            )

    payload = {
        "source_file": source_path.name,
        "edf_file": output_path.name,
        "sampling_rate_hz": sampling_rate,
        "sample_count": sample_count,
        "channel_labels": channel_labels,
        "start_time": start_iso,
        "events": event_payload,
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(payload, indent=2))


def convert_file(
    input_path: Path,
    output_dir: Path,
    patient_rules: list[dict[str, str]],
    *,
    output_path: Path | None = None,
    input_root: Path | None = None,
    json_sidecar: bool = False,
    resample_to: float | None = None,
    lowcut: float | None = None,
    highcut: float | None = None,
    notch: float | None = None,
    status_cb=None,
) -> Path:
    if status_cb:
        status_cb("read header")
    public_header, nrv_header = read_nervus_header(input_path)
    fs = public_header.get("Fs") or nrv_header.targetSamplingRate
    if not fs:
        raise RuntimeError("Unable to determine sampling frequency from header")

    channels = _select_channels(nrv_header)
    if not channels:
        raise RuntimeError("No channels available for conversion")

    if status_cb:
        status_cb("read data")
    channel_labels = _channel_labels(nrv_header, channels)
    waveform = read_nervus_data(input_path, nrv_header, channels=channels).T  # samples x channels

    # Apply optional bandpass/highpass/lowpass filter
    if lowcut is not None or highcut is not None:
        if status_cb:
            status_cb("bandpass filter")
        waveform = _bandpass_filter(waveform, float(fs), lowcut, highcut)

    # Apply optional notch filter (powerline removal)
    if notch is not None and notch > 0:
        if status_cb:
            status_cb("notch filter")
        waveform = _notch_filter(waveform, float(fs), notch)

    if resample_to is not None:
        if status_cb:
            status_cb("resample")
        waveform = _resample_waveform(waveform, float(fs), float(resample_to))
        fs = float(resample_to)

    patient_meta = _select_patient_metadata(input_path, patient_rules)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = Path(output_path) if output_path else _candidate_output_path(input_path, output_dir, input_root)
    if status_cb:
        status_cb("write edf")
    write_edf(
        resolved_output,
        waveform,
        fs,
        channel_labels,
        patient_meta,
        recording_start=nrv_header.startDateTime,
        annotations=nrv_header.Events,
    )
    if json_sidecar:
        if status_cb:
            status_cb("write json")
        _write_json_sidecar(
            resolved_output,
            source_path=input_path,
            sampling_rate=fs,
            channel_labels=channel_labels,
            sample_count=waveform.shape[0],
            start_time=nrv_header.startDateTime,
            events=nrv_header.Events,
        )
    logger.info(
        "Converted %s → %s (%d channels @ %.2f Hz)",
        input_path.name,
        resolved_output.name,
        len(channel_labels),
        fs,
    )
    return resolved_output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "ui", False):
        if not rich_available():
            print(
                "UI mode requires 'rich'.\n\n"
                "Install it (pick one):\n"
                "- uv run --with rich nicolet-e2edf --ui\n"
                "- uv pip install -p .venv rich && nicolet-e2edf --ui\n"
                "- uv pip install -p .venv '.[tui]' && nicolet-e2edf --ui\n"
            )
            return 1
        args = run_rich_wizard(title="nicolet-e2edf")

    if not args.input_paths:
        parser.error("--in is required (or run the interactive UI with --ui)")
    if not args.output_dir:
        parser.error("--out is required (or run the interactive UI with --ui)")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    input_paths = [Path(raw).expanduser() for raw in args.input_paths]
    output_dir = Path(args.output_dir).expanduser()
    patient_rules = _load_patient_rules(args.patient_json)

    try:
        inputs_with_roots = _discover_inputs_with_roots(input_paths, args.glob)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not inputs_with_roots:
        logger.error("No input files matched the supplied path/pattern")
        return 1

    candidate_outputs: dict[tuple[Path, Path | None], Path] = {}
    collisions: dict[Path, list[tuple[Path, Path | None]]] = {}
    for source_path, source_root in inputs_with_roots:
        candidate = _candidate_output_path(source_path, output_dir, source_root)
        key = (source_path, source_root)
        candidate_outputs[key] = candidate
        collisions.setdefault(candidate, []).append(key)

    resolved_outputs: dict[tuple[Path, Path | None], Path] = {}
    for candidate, sources in collisions.items():
        if len(sources) == 1:
            resolved_outputs[sources[0]] = candidate
            continue
        for source_path, source_root in sources:
            digest = hashlib.sha1(
                str(source_path).encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:8].upper()
            resolved_outputs[(source_path, source_root)] = candidate.with_name(
                f"{candidate.stem}_{digest}{candidate.suffix}"
            )

    if getattr(args, "ui", False):
        tui_inputs = [
            (source_path, source_root, resolved_outputs[(source_path, source_root)])
            for source_path, source_root in inputs_with_roots
        ]

        def _convert_one(*, source_path: Path, output_path: Path, input_root: Path | None, status_cb) -> None:
            convert_file(
                source_path,
                output_dir,
                patient_rules,
                output_path=output_path,
                input_root=input_root,
                json_sidecar=args.json_sidecar,
                resample_to=args.resample_to,
                lowcut=getattr(args, "lowcut", None),
                highcut=getattr(args, "highcut", None),
                notch=getattr(args, "notch", None),
                status_cb=status_cb,
            )

        return run_tui(
            inputs=tui_inputs,
            options=TuiOptions(
                json_sidecar=bool(getattr(args, "json_sidecar", False)),
                resample_to=getattr(args, "resample_to", None),
                lowcut=getattr(args, "lowcut", None),
                highcut=getattr(args, "highcut", None),
                notch=getattr(args, "notch", None),
                verbose=bool(getattr(args, "verbose", False)),
            ),
            convert_one=_convert_one,
            title="nicolet-e2edf",
        )

    success = 0
    for source_path, source_root in inputs_with_roots:
        try:
            convert_file(
                source_path,
                output_dir,
                patient_rules,
                output_path=resolved_outputs[(source_path, source_root)],
                input_root=source_root,
                json_sidecar=args.json_sidecar,
                resample_to=args.resample_to,
                lowcut=args.lowcut,
                highcut=args.highcut,
                notch=args.notch,
            )
            success += 1
        except Exception as exc:  # pragma: no cover - logged for CLI feedback
            logger.error("Failed to convert %s: %s", source_path, exc)
    if success == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

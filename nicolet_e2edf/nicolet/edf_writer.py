from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np


def _clean_ascii(text: str | None, fallback: str, length: int) -> str:
    raw = (text or fallback).encode("ascii", "ignore")[:length]
    cleaned = raw.decode("ascii", errors="ignore").strip()
    return cleaned or fallback[:length]


def _pad(text: str, length: int) -> bytes:
    data = text.encode("ascii", "ignore")[:length]
    return data.ljust(length, b" ")


def _format_float(value: float, length: int, decimals: int = 3) -> bytes:
    text = f"{value:.{decimals}f}"
    if len(text) > length:
        text = f"{value:.1f}"
    if len(text) > length:
        text = f"{int(value)}"
    return _pad(text, length)


def _channel_label(name: str, index: int) -> str:
    cleaned = name.split("\x00", 1)[0].strip()
    cleaned = "".join(ch for ch in cleaned if 32 <= ord(ch) <= 126)
    if not cleaned:
        cleaned = f"Ch{index}"
    return cleaned[:16]


def write_edf(
    output_path: str | Path,
    data_uV: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    patient_meta: dict[str, str] | None = None,
    recording_start: datetime | None = None,
) -> Path:
    """Write an EDF+ compatible file from microvolt data."""

    if data_uV.ndim != 2:
        raise ValueError("EDF writer expects data shaped as [samples, channels]")
    n_samples, n_channels = data_uV.shape
    if n_channels != len(ch_names):
        raise ValueError("Channel-name list must match waveform column count")
    if n_samples == 0 or sfreq <= 0:
        raise ValueError("Sampling frequency and data must be non-zero")

    patient_meta = patient_meta or {}
    start = recording_start or datetime.utcnow()
    patient_field = _clean_ascii(patient_meta.get("PatientName"), "Anon", 80)
    recording_field = _clean_ascii(
        patient_meta.get("StudyDescription") or patient_meta.get("SeriesDescription"),
        "nicolet-e2edf export",
        80,
    )

    header_bytes = 256 + n_channels * 256
    record_duration = n_samples / sfreq
    number_of_records = 1
    start_date = start.strftime("%d.%m.%y")
    start_time = start.strftime("%H.%M.%S")

    header = bytearray()
    header.extend(_pad("0", 8))
    header.extend(_pad(patient_field, 80))
    header.extend(_pad(recording_field, 80))
    header.extend(_pad(start_date, 8))
    header.extend(_pad(start_time, 8))
    header.extend(_pad(str(header_bytes), 8))
    header.extend(_pad("nicolet-e2edf", 44))
    header.extend(_pad(str(number_of_records), 8))
    header.extend(_format_float(record_duration, 8, decimals=5))
    header.extend(_pad(str(n_channels), 4))

    labels = [_channel_label(name, idx + 1) for idx, name in enumerate(ch_names)]
    physical_min = np.min(data_uV, axis=0)
    physical_max = np.max(data_uV, axis=0)
    physical_diff = np.maximum(physical_max - physical_min, 1.0)
    digital_min = np.full(n_channels, -32768, dtype=np.int32)
    digital_max = np.full(n_channels, 32767, dtype=np.int32)

    sections = [
        (labels, lambda text: _pad(text, 16)),
        (["" for _ in range(n_channels)], lambda text: _pad(text, 80)),
        (["uV" for _ in range(n_channels)], lambda text: _pad(text, 8)),
        (physical_min, lambda value: _format_float(float(value), 8)),
        (physical_max, lambda value: _format_float(float(value), 8)),
        (digital_min, lambda value: _pad(str(int(value)), 8)),
        (digital_max, lambda value: _pad(str(int(value)), 8)),
        (["None" for _ in range(n_channels)], lambda text: _pad(text, 80)),
        ([n_samples for _ in range(n_channels)], lambda value: _pad(str(int(value)), 8)),
        (["" for _ in range(n_channels)], lambda text: _pad(text, 32)),
    ]

    for values, formatter in sections:
        for value in values:
            header.extend(formatter(value))

    digital_range = (digital_max - digital_min).astype(np.float64)
    scales = digital_range / physical_diff.astype(np.float64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(header)
        for ch_idx in range(n_channels):
            scaled = (
                (data_uV[:, ch_idx] - physical_min[ch_idx]) * scales[ch_idx] + digital_min[ch_idx]
            )
            signal = np.clip(np.rint(scaled), -32768, 32767).astype("<i2")
            handle.write(signal.tobytes())
    return output_path

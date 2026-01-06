from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from .types import EventItem


# EDF+ specification month abbreviations (must be uppercase, exactly 3 chars)
_MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", 
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _clean_ascii(text: str | None, fallback: str, length: int) -> str:
    """Clean text to ASCII-only, returning fallback if empty."""
    raw = (text or fallback).encode("ascii", "ignore")[:length]
    cleaned = raw.decode("ascii", errors="ignore").strip()
    return cleaned or fallback[:length]


def _sanitize_subfield(text: str) -> str:
    """Sanitize a subfield for EDF+ structured fields.
    
    EDF+ uses spaces as delimiters in patient/recording fields.
    Subfields containing spaces must use underscores instead.
    Only printable ASCII 32-126 allowed, but spaces become underscores.
    """
    # Replace spaces with underscores per EDF+ spec
    text = text.replace(" ", "_")
    # Keep only printable ASCII (32-126), but we already replaced spaces
    cleaned = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    return cleaned or "X"


def _format_edfplus_patient(patient_meta: dict[str, str]) -> str:
    """Format patient identification per EDF+ spec.
    
    EDF+ requires: <patient_code> <sex> <birthdate> <patient_name>
    - Spaces are used as delimiters between subfields
    - Subfields use underscores for internal spaces
    - Unknown fields should use 'X'
    - Sex: M, F, or X (unknown)
    - Birthdate: DD-MMM-YYYY (e.g., 02-MAY-1951) or X
    """
    # Patient code (ID)
    patient_code = _sanitize_subfield(patient_meta.get("PatientID", "X"))
    
    # Sex: M, F, or X
    sex_raw = patient_meta.get("PatientSex", "X").upper().strip()
    if sex_raw in ("M", "MALE"):
        sex = "M"
    elif sex_raw in ("F", "FEMALE"):
        sex = "F"
    else:
        sex = "X"
    
    # Birthdate: DD-MMM-YYYY or X
    birthdate = "X"
    birthdate_raw = patient_meta.get("PatientBirthDate", "")
    if birthdate_raw:
        # Try to parse common date formats
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y%m%d"):
            try:
                dt = datetime.strptime(birthdate_raw, fmt)
                birthdate = f"{dt.day:02d}-{_MONTH_ABBR[dt.month - 1]}-{dt.year}"
                break
            except ValueError:
                continue
    
    # Patient name (use underscores for spaces)
    patient_name = _sanitize_subfield(patient_meta.get("PatientName", "X"))
    
    # Combine with single spaces as delimiters
    return f"{patient_code} {sex} {birthdate} {patient_name}"


def _format_edfplus_recording(recording_start: datetime, patient_meta: dict[str, str]) -> str:
    """Format recording identification per EDF+ spec.
    
    EDF+ requires: Startdate <DD-MMM-YYYY> <admin_code> <technician> <equipment>
    - Must start with literal "Startdate " followed by the date
    - Date format: DD-MMM-YYYY (e.g., 02-MAR-2002)
    - Unknown subfields use 'X'
    """
    # Format date as DD-MMM-YYYY
    date_str = f"{recording_start.day:02d}-{_MONTH_ABBR[recording_start.month - 1]}-{recording_start.year}"
    
    # Admin code (use study description or X)
    admin_code = _sanitize_subfield(
        patient_meta.get("StudyDescription") or 
        patient_meta.get("SeriesDescription") or 
        "X"
    )
    
    # Technician (unknown)
    technician = "X"
    
    # Equipment identifier
    equipment = "nicolet-e2edf"
    
    return f"Startdate {date_str} {admin_code} {technician} {equipment}"


def _pad(text: str, length: int) -> bytes:
    data = text.encode("ascii", "ignore")[:length]
    return data.ljust(length, b" ")


def _format_float(value: float, length: int, decimals: int = 3) -> bytes:
    """Format a float for EDF header fields.
    
    EDF spec requires ASCII representation within fixed width.
    We prefer simpler integer representation when the value is a whole number.
    """
    # If the value is effectively an integer, write it as such
    if value == int(value):
        text = str(int(value))
        if len(text) <= length:
            return _pad(text, length)
    
    # Otherwise, try with decreasing precision
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


def _format_annotation_events(
    events: Sequence[EventItem] | None,
    recording_start: datetime,
) -> tuple[np.ndarray, int]:
    """Return EDF+ annotation signal and samples-per-record for the channel.

    The returned array contains the EDF+ Time-Annotation Lists (TALs) encoded
    as raw bytes within int16 samples. Per EDF+ spec section 2.2.4:
    - Each data record MUST start with a time-keeping TAL: +<onset>\x14\x14\x00
    - This TAL has no annotations, just the time stamp for the data record
    - Additional TALs with annotations follow after the time-keeper
    
    The TAL bytes are stored directly as raw bytes within int16 samples.
    Each int16 sample holds 2 bytes of TAL data (or zero-padding).
    """
    tal_segments: list[bytes] = []
    
    # EDF+ REQUIRES a time-keeping TAL at the start of each data record.
    # Format: +<onset>\x14\x14\x00 (onset with empty annotation)
    # For a single data record starting at time 0:
    time_keeper_tal = b"+0\x14\x14\x00"
    tal_segments.append(time_keeper_tal)
    
    # Add event annotations as additional TALs
    if events:
        for event in events:
            if event.date is None:
                continue
            onset_seconds = (event.date - recording_start).total_seconds()
            # Format onset with sign (+ or -) as required by EDF+
            onset = f"{onset_seconds:+.6f}"
            
            # Duration is optional, format: \x15<duration>
            duration = ""
            if event.duration is not None and event.duration > 0:
                duration = f"\x15{event.duration:.6f}"
            
            # Build annotation text
            description_parts = []
            if event.label:
                description_parts.append(event.label)
            if event.annotation:
                description_parts.append(event.annotation)
            elif event.IDStr and not description_parts:
                description_parts.append(event.IDStr)
            text = ": ".join(part for part in description_parts if part) or "Event"
            
            # TAL format: <onset><duration>\x14<annotation>\x14\x00
            tal_segments.append(f"{onset}{duration}\x14{text}\x14\x00".encode("ascii", "ignore"))

    # Combine all TALs into a single byte stream
    tal_stream = b"".join(tal_segments)
    
    # Calculate number of int16 samples needed to store the bytes
    # Each int16 sample stores 2 bytes of TAL data
    # We need ceiling division: (len + 1) // 2
    byte_count = len(tal_stream)
    sample_count = (byte_count + 1) // 2  # Round up to fit all bytes
    sample_count = max(sample_count, 1)   # At least 1 sample
    
    # Pad the byte stream to even length for int16 alignment
    if byte_count % 2 != 0:
        tal_stream = tal_stream + b'\x00'
    
    # Store TAL bytes directly as raw bytes within int16 array
    # The bytes are packed 2 per int16 sample (little-endian)
    signal = np.zeros(sample_count, dtype=np.int16)
    signal[:len(tal_stream) // 2] = np.frombuffer(tal_stream, dtype='<i2')
    
    return signal, sample_count


def write_edf(
    output_path: str | Path,
    data_uV: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    patient_meta: dict[str, str] | None = None,
    recording_start: datetime | None = None,
    annotations: Sequence[EventItem] | None = None,
) -> Path:
    """Write an EDF+ compatible file from microvolt data.
    
    Outputs files compliant with the EDF+ specification:
    https://www.edfplus.info/specs/edfplus.html
    
    Key EDF+ compliance features:
    - Reserved field starts with 'EDF+C' (continuous recording)
    - Patient identification uses structured format: code sex birthdate name
    - Recording identification starts with 'Startdate DD-MMM-YYYY'
    - EDF Annotations signal for events with time-keeping TAL
    """
    if data_uV.ndim != 2:
        raise ValueError("EDF writer expects data shaped as [samples, channels]")
    n_samples, n_channels = data_uV.shape
    if n_channels != len(ch_names):
        raise ValueError("Channel-name list must match waveform column count")
    if n_samples == 0 or sfreq <= 0:
        raise ValueError("Sampling frequency and data must be non-zero")

    patient_meta = patient_meta or {}
    start = recording_start or datetime.utcnow()
    
    # EDF+ compliant patient identification field
    # Format: <code> <sex> <birthdate> <name> with underscores for spaces in subfields
    patient_field = _format_edfplus_patient(patient_meta)
    
    # EDF+ compliant recording identification field
    # Format: Startdate DD-MMM-YYYY <admin_code> <technician> <equipment>
    recording_field = _format_edfplus_recording(start, patient_meta)

    include_annotations = annotations is not None
    annotation_signal = None
    annotation_samples = 0
    if include_annotations:
        annotation_signal, annotation_samples = _format_annotation_events(annotations, start)

    total_channels = n_channels + (1 if include_annotations else 0)
    header_bytes = 256 + total_channels * 256
    record_duration = n_samples / sfreq
    number_of_records = 1
    
    # Date format: DD.MM.YY per EDF spec
    # Note: 2-digit year uses 1985 clipping (85-99 = 1985-1999, 00-84 = 2000-2084)
    start_date = start.strftime("%d.%m.%y")
    start_time = start.strftime("%H.%M.%S")

    # Build the 256-byte fixed header
    header = bytearray()
    
    # Version: 8 bytes, must be "0" followed by spaces
    header.extend(_pad("0", 8))
    
    # Patient identification: 80 bytes, EDF+ structured format
    header.extend(_pad(patient_field, 80))
    
    # Recording identification: 80 bytes, EDF+ structured format
    header.extend(_pad(recording_field, 80))
    
    # Start date: 8 bytes, DD.MM.YY
    header.extend(_pad(start_date, 8))
    
    # Start time: 8 bytes, HH.MM.SS
    header.extend(_pad(start_time, 8))
    
    # Number of bytes in header: 8 bytes
    header.extend(_pad(str(header_bytes), 8))
    
    # Reserved: 44 bytes
    # EDF+ REQUIRES this field to start with "EDF+C" (continuous) or "EDF+D" (discontinuous)
    # Only mark as EDF+ if we're including the EDF Annotations signal
    # Without annotations, we write plain EDF (empty reserved field)
    if include_annotations:
        header.extend(_pad("EDF+C", 44))
    else:
        header.extend(_pad("", 44))
    
    # Number of data records: 8 bytes
    header.extend(_pad(str(number_of_records), 8))
    
    # Duration of data record in seconds: 8 bytes
    header.extend(_format_float(record_duration, 8, decimals=5))
    
    # Number of signals: 4 bytes
    header.extend(_pad(str(total_channels), 4))

    # Build signal labels
    labels = [_channel_label(name, idx + 1) for idx, name in enumerate(ch_names)]
    
    # Calculate physical ranges from actual data
    physical_min = np.min(data_uV, axis=0)
    physical_max = np.max(data_uV, axis=0)
    
    # EDF spec requires physical_min < physical_max (strictly less than)
    # For constant signals (e.g., all zeros), ensure a minimal range
    # We add a tiny offset to max when min == max
    equal_mask = physical_min == physical_max
    if np.any(equal_mask):
        # Add 1.0 to max for channels with constant values (preserves the data)
        physical_max = np.where(equal_mask, physical_max + 1.0, physical_max)
    
    physical_diff = np.maximum(physical_max - physical_min, 1.0)
    
    # Digital range for EEG signals: full 16-bit signed range
    digital_min = np.full(n_channels, -32768, dtype=np.int32)
    digital_max = np.full(n_channels, 32767, dtype=np.int32)
    samples_per_record = np.full(n_channels, n_samples, dtype=np.int32)

    if include_annotations and annotation_signal is not None:
        # EDF Annotations signal per EDF+ spec section 2.2.1
        labels.append("EDF Annotations")
        
        # EDF+ spec: annotation signal uses physical range -1 to 1
        # and digital range -32768 to 32767
        # This allows TAL bytes (0-127 ASCII) to be stored correctly
        physical_min = np.concatenate([physical_min, np.array([-1.0])])
        physical_max = np.concatenate([physical_max, np.array([1.0])])
        physical_diff = np.concatenate([physical_diff, np.array([2.0])])  # 1 - (-1) = 2
        digital_min = np.concatenate([digital_min, np.array([-32768], dtype=np.int32)])
        digital_max = np.concatenate([digital_max, np.array([32767], dtype=np.int32)])
        samples_per_record = np.concatenate([samples_per_record, np.array([annotation_samples])])

    # Build signal header sections per EDF spec
    # Each section contains one field for each signal, stored sequentially
    sections = [
        # Signal labels (16 bytes each)
        (labels, lambda text: _pad(text, 16)),
        # Transducer type (80 bytes each) - empty for now
        (["" for _ in range(total_channels)], lambda text: _pad(text, 80)),
        # Physical dimension (8 bytes each)
        (["uV" for _ in range(n_channels)] + ["" for _ in range(total_channels - n_channels)], lambda text: _pad(text, 8)),
        # Physical minimum (8 bytes each)
        (physical_min, lambda value: _format_float(float(value), 8)),
        # Physical maximum (8 bytes each)
        (physical_max, lambda value: _format_float(float(value), 8)),
        # Digital minimum (8 bytes each)
        (digital_min, lambda value: _pad(str(int(value)), 8)),
        # Digital maximum (8 bytes each)
        (digital_max, lambda value: _pad(str(int(value)), 8)),
        # Prefiltering info (80 bytes each) - empty per EDF spec (can be filled if known)
        (["" for _ in range(total_channels)], lambda text: _pad(text, 80)),
        # Number of samples per data record (8 bytes each)
        (samples_per_record, lambda value: _pad(str(int(value)), 8)),
        # Reserved (32 bytes each) - must be empty
        (["" for _ in range(total_channels)], lambda text: _pad(text, 32)),
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
        for ch_idx in range(total_channels):
            if include_annotations and ch_idx == total_channels - 1:
                signal = annotation_signal.astype("<i2") if annotation_signal is not None else np.array([], dtype="<i2")
            else:
                scaled = (
                    (data_uV[:, ch_idx] - physical_min[ch_idx]) * scales[ch_idx] + digital_min[ch_idx]
                )
                signal = np.clip(np.rint(scaled), -32768, 32767).astype("<i2")
            handle.write(signal.tobytes())
    return output_path

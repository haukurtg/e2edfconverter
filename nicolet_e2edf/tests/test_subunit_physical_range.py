"""Regression tests for sub-µV EDF physical-range quantisation.

Channels whose whole-recording physical range is below 1 µV must keep that
range in the header and use the serialised header bounds for scaling. Without
the fix, the writer clamped the scale to max(range, 1.0) while storing the
true min/max, which squashed decoded amplitudes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nicolet_e2edf.nicolet.edf_writer import write_edf
from nicolet_e2edf.nicolet.types import EventItem

# pyedflib is already a declared optional/dev dependency used by other EDF tests
pyedflib = pytest.importorskip("pyedflib", reason="pyedflib required for sub-µV round-trip tests")


def _header_physical_bounds(payload: bytes, n_signals: int) -> list[tuple[float, float]]:
    """Parse physical_min/physical_max from the serialised EDF header bytes."""
    # Fixed header, then labels / transducers / dimensions, then phys min/max.
    offset = 256 + n_signals * (16 + 80 + 8)
    bounds: list[tuple[float, float]] = []
    for index in range(n_signals):
        min_start = offset + index * 8
        max_start = offset + n_signals * 8 + index * 8
        physical_min = float(payload[min_start : min_start + 8].decode("ascii").strip())
        physical_max = float(payload[max_start : max_start + 8].decode("ascii").strip())
        bounds.append((physical_min, physical_max))
    return bounds


def _decode_int16_channel(
    digital: np.ndarray,
    physical_min: float,
    physical_max: float,
) -> tuple[np.ndarray, float]:
    """Decode EDF int16 samples using header physical bounds; return samples and LSB."""
    lsb = (physical_max - physical_min) / 65535.0
    decoded = (digital.astype(np.float64) + 32768.0) * lsb + physical_min
    return decoded, lsb


def _assert_within_half_lsb(
    source: np.ndarray,
    decoded: np.ndarray,
    lsb: float,
) -> None:
    """Each sample must match the source within half an EDF LSB."""
    assert np.max(np.abs(decoded - source)) <= 0.5 * lsb + 1e-12


@pytest.mark.parametrize(
    "physical_range",
    [0.023, 0.3, 0.99],
    ids=["range-0.023", "range-0.3", "range-0.99"],
)
def test_subunit_uv_round_trip_within_half_lsb(tmp_path: Path, physical_range: float) -> None:
    """Sub-µV channels must round-trip within half an LSB of the header range."""
    # Linear ramp across the full channel range (units are µV).
    samples = np.linspace(0.0, physical_range, 8, dtype=np.float64).reshape(-1, 1)
    output = tmp_path / f"subunit_{physical_range}.edf"
    write_edf(output, samples, 8.0, ["SubUV"])

    with pyedflib.EdfReader(str(output)) as reader:
        decoded = reader.readSignal(0)
        physical_min = reader.getPhysicalMinimum(0)
        physical_max = reader.getPhysicalMaximum(0)

    # Header must preserve a true sub-µV span, not the old 1.0 clamp.
    assert physical_max - physical_min < 1.0
    assert physical_max - physical_min >= physical_range - 1e-9

    lsb = (physical_max - physical_min) / 65535.0
    _assert_within_half_lsb(samples[:, 0], decoded, lsb)


def test_quantisation_uses_serialised_header_bounds(tmp_path: Path) -> None:
    """Scale against the ASCII header floats, including inward-rounding extrema."""
    # 12.34567 formats to 3 decimals as 12.346 (outward for a max).
    # 10.0004 formats to 10.000, which is inward for a max and must be nudged out.
    data = np.array(
        [
            [10.0004],
            [11.0],
            [12.34567],
            [10.5],
        ],
        dtype=np.float64,
    )
    output = tmp_path / "header_bounds.edf"
    write_edf(output, data, 4.0, ["C1"])

    payload = output.read_bytes()
    physical_min, physical_max = _header_physical_bounds(payload, n_signals=1)[0]

    # Bounds come from the header bytes, not from the NumPy extrema.
    assert physical_min == float(payload[360:368].decode("ascii").strip())
    assert physical_max == float(payload[368:376].decode("ascii").strip())
    # The serialised max must still cover the true source maximum.
    assert physical_max >= float(np.max(data))
    assert physical_min <= float(np.min(data))

    header_bytes = 256 + 256
    digital = np.frombuffer(payload[header_bytes : header_bytes + 8], dtype="<i2")
    decoded, lsb = _decode_int16_channel(digital, physical_min, physical_max)
    _assert_within_half_lsb(data[:, 0], decoded, lsb)


def test_quantisation_matches_large_non_integer_extrema(tmp_path: Path) -> None:
    """Header/serialisation mismatch used to exceed half an LSB on large values."""
    data = np.array(
        [[-14097.13603971], [-13605.90261275], [-13690.65661762], [-14042.91978912]]
    )
    output = tmp_path / "large_extrema.edf"
    write_edf(output, data, 4.0, ["C1"])

    payload = output.read_bytes()
    physical_min, physical_max = _header_physical_bounds(payload, n_signals=1)[0]
    digital = np.frombuffer(payload[512:520], dtype="<i2")
    decoded, lsb = _decode_int16_channel(digital, physical_min, physical_max)
    _assert_within_half_lsb(data[:, 0], decoded, lsb)


def test_normal_range_channels_still_decode(tmp_path: Path) -> None:
    """Typical ±200 µV EEG channels remain within half an LSB after the fix."""
    samples = np.linspace(-200.0, 200.0, 16, dtype=np.float64).reshape(-1, 1)
    output = tmp_path / "normal_range.edf"
    write_edf(output, samples, 16.0, ["Cz"])

    with pyedflib.EdfReader(str(output)) as reader:
        decoded = reader.readSignal(0)
        lsb = (reader.getPhysicalMaximum(0) - reader.getPhysicalMinimum(0)) / 65535.0

    _assert_within_half_lsb(samples[:, 0], decoded, lsb)


def test_constant_channel_still_decodes(tmp_path: Path) -> None:
    """Constant channels keep the forced +1.0 range and decode correctly."""
    samples = np.full((8, 1), 12.5, dtype=np.float64)
    output = tmp_path / "constant.edf"
    write_edf(output, samples, 8.0, ["Const"])

    payload = output.read_bytes()
    physical_min, physical_max = _header_physical_bounds(payload, n_signals=1)[0]
    assert physical_max - physical_min == 1.0

    with pyedflib.EdfReader(str(output)) as reader:
        decoded = reader.readSignal(0)
        lsb = (reader.getPhysicalMaximum(0) - reader.getPhysicalMinimum(0)) / 65535.0

    _assert_within_half_lsb(samples[:, 0], decoded, lsb)


def test_annotation_channel_header_and_bytes_unchanged(tmp_path: Path) -> None:
    """EDF Annotations stay at physical -1/1 with unchanged TAL payload bytes."""
    data = np.array([[100.0], [-50.0], [200.0], [0.0]], dtype=np.float64)
    start = datetime(2020, 1, 2, 3, 4, 5)
    events = [
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=start,
            duration=0.0,
            user="test",
            GUID="{TEST}",
            label="TestEvent",
            IDStr="Event",
            annotation="hello",
        )
    ]
    output = tmp_path / "annotations.edf"
    write_edf(
        output,
        data,
        4.0,
        ["C3"],
        recording_start=start,
        annotations=events,
    )

    payload = output.read_bytes()
    bounds = _header_physical_bounds(payload, n_signals=2)
    assert bounds[1] == (-1.0, 1.0)

    # Golden annotation-channel bytes captured from unpatched public main (v0.4.2).
    # One data record: 4 EEG int16 samples (8 bytes) then 17 annotation samples.
    header_bytes = 256 + 2 * 256
    annotation_bytes = payload[header_bytes + 8 :]
    assert annotation_bytes.hex() == (
        "2b301414002b302e30303030303014546573744576656e743a2068656c6c6f140000"
    )

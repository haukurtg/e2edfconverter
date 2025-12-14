from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nicolet_e2edf.nicolet import cli
from nicolet_e2edf.nicolet.types import EventItem, NervusHeader, SegmentInfo


def test_convert_to_edf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_header = NervusHeader(filename=tmp_path / "case.e")
    fake_header.matchingChannels = [1]
    fake_header.targetSamplingRate = 128.0
    fake_header.Segments = [
        SegmentInfo(
            dateOLE=0.0,
            date=datetime(2021, 5, 5, 8, 30, 0),
            duration=4 / 128.0,
            chName=["C3"],
            refName=["REF"],
            samplingRate=np.array([128.0]),
            scale=np.ones(1),
            sampleCount=np.array([4]),
        )
    ]
    fake_header.startDateTime = datetime(2021, 5, 5, 8, 30, 0)
    fake_header.Events = [
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2021, 5, 5, 8, 30, 1),
            duration=2.0,
            user="user",
            GUID="{GUID}",
            label="TestEvent",
            IDStr="TestEvent",
            annotation="note",
        )
    ]

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(recording), "--out", str(output_dir)])
    assert exit_code == 0

    edf_path = output_dir / "case.edf"
    assert edf_path.exists()

    content = edf_path.read_bytes()
    header = content[:256 + 256 * 2]  # base + two signals
    n_signals = int(header[252:256].decode("ascii").strip())
    assert n_signals == 2

    label_section = header[256 : 256 + 32]
    assert label_section[:16].decode("ascii").strip() == "C3"
    assert label_section[16:32].decode("ascii").strip() == "EDF Annotations"

    samples_offset = 256
    samples_offset += 16 * n_signals  # labels
    samples_offset += 80 * n_signals  # transducer
    samples_offset += 8 * n_signals  # physical dimension
    samples_offset += 8 * n_signals  # physical min
    samples_offset += 8 * n_signals  # physical max
    samples_offset += 8 * n_signals  # digital min
    samples_offset += 8 * n_signals  # digital max
    samples_offset += 80 * n_signals  # prefilter
    samples_section = header[samples_offset : samples_offset + 8 * n_signals]
    samples_counts = [
        int(samples_section[i * 8 : (i + 1) * 8].decode("ascii").strip()) for i in range(n_signals)
    ]
    assert samples_counts[0] == 4

    tal_bytes = (
        "+1.000000".encode("ascii")
        + b"\x152.000000"
        + b"\x14"
        + "TestEvent: note".encode("ascii")
        + b"\x14\x00"
    )
    assert samples_counts[1] >= len(tal_bytes)

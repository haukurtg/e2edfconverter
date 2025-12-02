from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nicolet_e2edf.nicolet import cli
from nicolet_e2edf.nicolet.types import NervusHeader, SegmentInfo


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
    header = content[:256 + 256]  # base + one signal
    n_signals = int(header[252:256].decode("ascii").strip())
    assert n_signals == 1

    label_section = header[256 : 256 + 16]
    assert label_section.decode("ascii").strip() == "C3"

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
    samples_per_record = int(samples_section.decode("ascii").strip())
    assert samples_per_record == 4

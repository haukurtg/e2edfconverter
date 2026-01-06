from __future__ import annotations

import json
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
    
    # With 1-second data records, samples_per_record = sampling frequency
    # The actual data (4 samples) is padded to fill the 1-second record
    assert samples_counts[0] == 128  # samples per 1-second record at 128 Hz

    # Verify annotation signal has enough samples for TAL data
    # The annotation samples are sized to hold all events for the worst-case record
    assert samples_counts[1] >= 8  # minimum annotation samples


def test_resample_and_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        return np.array([[0, 10, 20, 30]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(
        ["--in", str(recording), "--out", str(output_dir), "--json-sidecar", "--resample-to", "64"]
    )
    assert exit_code == 0

    edf_path = output_dir / "case.edf"
    sidecar_path = output_dir / "case.json"
    assert edf_path.exists()
    assert sidecar_path.exists()

    content = edf_path.read_bytes()
    header = content[:256 + 256 * 2]
    samples_offset = 256
    n_signals = int(header[252:256].decode("ascii").strip())
    samples_offset += 16 * n_signals
    samples_offset += 80 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 8 * n_signals
    samples_offset += 80 * n_signals
    samples_section = header[samples_offset : samples_offset + 8 * n_signals]
    samples_counts = [
        int(samples_section[i * 8 : (i + 1) * 8].decode("ascii").strip()) for i in range(n_signals)
    ]
    
    # With 1-second data records, samples_per_record = resampled frequency
    # The actual data (2 samples after resampling to 64 Hz) is padded to fill 1-second record
    assert samples_counts[0] == 64  # samples per 1-second record at 64 Hz

    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["sampling_rate_hz"] == 64
    # sample_count in sidecar is the ACTUAL sample count, not padded
    assert sidecar["sample_count"] == 2
    # Events with annotation text go to "annotations" list, not "events"
    assert sidecar["annotations"][0]["onset_seconds"] == 1.0


def test_directory_input_preserves_subfolders_and_avoids_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    input_root = tmp_path / "inputs_nested"
    (input_root / "A").mkdir(parents=True)
    (input_root / "B").mkdir(parents=True)
    (input_root / "A" / "same.e").write_bytes(b"\x00")
    (input_root / "B" / "same.e").write_bytes(b"\x00")

    output_dir = tmp_path / "out_nested"
    exit_code = cli.main(["--in", str(input_root), "--out", str(output_dir), "--glob", "**/*.e"])
    assert exit_code == 0

    assert (output_dir / "A" / "same.edf").exists()
    assert (output_dir / "B" / "same.edf").exists()

    input_root_flat = tmp_path / "inputs_flat"
    input_root_flat.mkdir()
    (input_root_flat / "case.e").write_bytes(b"\x00")
    (input_root_flat / "case.eeg").write_bytes(b"\x00")

    output_dir_flat = tmp_path / "out_flat"
    exit_code = cli.main(["--in", str(input_root_flat), "--out", str(output_dir_flat)])
    assert exit_code == 0

    case_outputs = list(output_dir_flat.glob("case*.edf"))
    assert len(case_outputs) == 2
    assert case_outputs[0].name != case_outputs[1].name


def test_multi_input_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    input_a = tmp_path / "a.e"
    input_b = tmp_path / "b.e"
    input_a.write_bytes(b"\x00")
    input_b.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    exit_code = cli.main(["--in", str(input_a), str(input_b), "--out", str(output_dir)])
    assert exit_code == 0
    assert (output_dir / "a.edf").exists()
    assert (output_dir / "b.edf").exists()


def test_ui_single_file_without_real_rich(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    fake_header.Events = []

    def _fake_read_header(path: Path):
        return {"Fs": 128.0}, fake_header

    def _fake_read_data(path: Path, header: NervusHeader, channels=None, begsample=None, endsample=None):
        return np.array([[10, 20, 30, 40]], dtype=np.float32)

    monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
    monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

    recording = tmp_path / "case.e"
    recording.write_bytes(b"\x00")
    output_dir = tmp_path / "out"

    def _fake_run_rich_wizard(*, title: str):
        return type(
            "Args",
            (),
            dict(
                input_paths=[str(recording)],
                output_dir=str(output_dir),
                glob="*.e",
                patient_json=None,
                json_sidecar=False,
                resample_to=None,
                verbose=False,
                ui=True,
            ),
        )()

    def _fake_run_tui(*, inputs, options, convert_one, title: str) -> int:
        for source_path, input_root, output_path in inputs:
            convert_one(source_path=source_path, output_path=output_path, input_root=input_root, status_cb=lambda _: None)
        return 0

    monkeypatch.setattr(cli, "rich_available", lambda: True)
    monkeypatch.setattr(cli, "run_rich_wizard", _fake_run_rich_wizard)
    monkeypatch.setattr(cli, "run_tui", _fake_run_tui)

    exit_code = cli.main(["--ui"])
    assert exit_code == 0
    assert (output_dir / "case.edf").exists()


def test_ui_requires_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "rich_available", lambda: False)
    exit_code = cli.main(["--ui"])
    assert exit_code == 1

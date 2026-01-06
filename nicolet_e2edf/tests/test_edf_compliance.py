"""EDF+ compliance validation tests using PyEDFlib.

PyEDFlib serves as a strict EDF+/BDF+ parser that will raise errors
if files don't comply with the EDF+ specification. This makes it
an excellent CI gate for ensuring our output is standards-compliant.

These tests verify that:
1. Generated EDF files can be opened by PyEDFlib without errors
2. Header fields are correctly formatted per EDF+ spec
3. Signal data can be read correctly
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nicolet_e2edf.nicolet import cli
from nicolet_e2edf.nicolet.edf_writer import write_edf
from nicolet_e2edf.nicolet.types import EventItem, NervusHeader, SegmentInfo

# PyEDFlib is an optional dev dependency - skip tests if not available
pyedflib = pytest.importorskip("pyedflib", reason="pyedflib required for compliance tests")


class TestEdfPlusCompliance:
    """Tests that verify EDF+ standard compliance using PyEDFlib."""

    def test_pyedflib_can_read_basic_edf(self, tmp_path: Path) -> None:
        """Verify PyEDFlib can read a basic EDF file without errors.
        
        This is the most fundamental compliance check - if PyEDFlib raises
        an error, the file is not EDF+ compliant.
        """
        # Create a simple test EDF file
        output_path = tmp_path / "test.edf"
        
        # Simple test data: 2 channels, 256 samples at 256 Hz (1 second)
        n_samples = 256
        n_channels = 2
        sfreq = 256.0
        data = np.random.randn(n_samples, n_channels) * 100  # Random EEG-like data in µV
        ch_names = ["Fp1", "Fp2"]
        
        patient_meta = {
            "PatientID": "TEST001",
            "PatientName": "Test Patient",
            "PatientSex": "M",
        }
        recording_start = datetime(2024, 3, 15, 10, 30, 0)
        
        # Write the EDF file
        write_edf(
            output_path,
            data,
            sfreq,
            ch_names,
            patient_meta=patient_meta,
            recording_start=recording_start,
            annotations=None,  # No annotations for this basic test
        )
        
        # This is the key test: PyEDFlib should read it without errors
        reader = pyedflib.EdfReader(str(output_path))
        try:
            # Verify basic header info
            assert reader.signals_in_file == n_channels
            assert reader.file_duration == pytest.approx(1.0, rel=0.01)
            
            # Verify we can read signal data
            for ch_idx in range(n_channels):
                signal = reader.readSignal(ch_idx)
                assert len(signal) == n_samples
        finally:
            reader.close()

    def test_pyedflib_can_read_edf_with_annotations(self, tmp_path: Path) -> None:
        """Verify PyEDFlib can read EDF+ files with EDF Annotations signal."""
        output_path = tmp_path / "test_annotations.edf"
        
        # Test data
        n_samples = 512
        n_channels = 3
        sfreq = 256.0
        data = np.random.randn(n_samples, n_channels) * 50
        ch_names = ["C3", "C4", "Cz"]
        
        patient_meta = {
            "PatientID": "ANNOT001",
            "PatientName": "Annotation Test",
            "PatientSex": "F",
        }
        recording_start = datetime(2024, 5, 20, 14, 0, 0)
        
        # Create some test events/annotations
        events = [
            EventItem(
                dateOLE=0.0,
                dateFraction=0.0,
                date=datetime(2024, 5, 20, 14, 0, 0),  # At recording start
                duration=0.0,
                user="test",
                GUID="{TEST}",
                label="RecordingStart",
                IDStr="Start",
                annotation="Recording started",
            ),
            EventItem(
                dateOLE=0.0,
                dateFraction=0.0,
                date=datetime(2024, 5, 20, 14, 0, 1),  # 1 second in
                duration=2.0,
                user="test",
                GUID="{TEST}",
                label="TestEvent",
                IDStr="Event",
                annotation="Test annotation at 1 second",
            ),
        ]
        
        # Write the EDF file with annotations
        write_edf(
            output_path,
            data,
            sfreq,
            ch_names,
            patient_meta=patient_meta,
            recording_start=recording_start,
            annotations=events,
        )
        
        # PyEDFlib should read it without errors
        reader = pyedflib.EdfReader(str(output_path))
        try:
            # PyEDFlib returns n_channels for EEG channels only
            # (annotation signal is handled internally)
            assert reader.signals_in_file == n_channels
            
            # Verify EEG data can still be read
            for ch_idx in range(n_channels):
                signal = reader.readSignal(ch_idx)
                assert len(signal) == n_samples
        finally:
            reader.close()

    def test_header_fields_comply_with_edfplus_spec(self, tmp_path: Path) -> None:
        """Verify header fields are formatted per EDF+ specification."""
        output_path = tmp_path / "test_header.edf"
        
        # Test data
        data = np.random.randn(128, 2) * 100
        ch_names = ["F3", "F4"]
        sfreq = 128.0
        
        patient_meta = {
            "PatientID": "P123",
            "PatientName": "John Doe",  # Name with space - should use underscore
            "PatientSex": "M",
            "PatientBirthDate": "1990-05-15",
        }
        recording_start = datetime(2024, 6, 10, 9, 15, 30)
        
        # Include annotations to create EDF+ format (with EDF+C marker)
        write_edf(
            output_path,
            data,
            sfreq,
            ch_names,
            patient_meta=patient_meta,
            recording_start=recording_start,
            annotations=[],  # Empty but present = EDF+ with annotation signal
        )
        
        # Read raw header bytes to verify format
        with open(output_path, "rb") as f:
            header = f.read(256)
        
        # Version field (8 bytes): must be "0" followed by spaces
        version = header[0:8].decode("ascii")
        assert version.strip() == "0"
        
        # Patient identification (80 bytes): EDF+ format
        patient_field = header[8:88].decode("ascii").strip()
        # Should be: <code> <sex> <birthdate> <name>
        parts = patient_field.split()
        assert len(parts) >= 4, f"Patient field not properly structured: {patient_field}"
        assert parts[1] == "M"  # Sex
        assert "MAY" in parts[2].upper()  # Birthdate contains month
        assert "John_Doe" in parts[3] or "John" in parts[3]  # Name with underscore
        
        # Recording identification (80 bytes): must start with "Startdate"
        recording_field = header[88:168].decode("ascii").strip()
        assert recording_field.startswith("Startdate"), f"Recording field: {recording_field}"
        # Should contain the date in DD-MMM-YYYY format
        assert "JUN" in recording_field.upper()  # June
        assert "2024" in recording_field
        
        # Reserved field (44 bytes at offset 192): must start with EDF+C or EDF+D
        reserved = header[192:236].decode("ascii").strip()
        assert reserved.startswith("EDF+C") or reserved.startswith("EDF+D"), \
            f"Reserved field must start with EDF+C or EDF+D, got: {reserved}"

    def test_edf_readable_after_conversion_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test full conversion pipeline produces PyEDFlib-readable files."""
        # Set up fake header/data like other tests
        fake_header = NervusHeader(filename=tmp_path / "case.e")
        fake_header.matchingChannels = [1, 2]
        fake_header.targetSamplingRate = 256.0
        fake_header.Segments = [
            SegmentInfo(
                dateOLE=0.0,
                date=datetime(2024, 1, 15, 10, 0, 0),
                duration=256 / 256.0,  # 1 second of data
                chName=["Fp1", "Fp2"],
                refName=["REF", "REF"],
                samplingRate=np.array([256.0, 256.0]),
                scale=np.ones(2),
                sampleCount=np.array([256, 256]),
            )
        ]
        fake_header.startDateTime = datetime(2024, 1, 15, 10, 0, 0)
        fake_header.Events = [
            EventItem(
                dateOLE=0.0,
                dateFraction=0.0,
                date=datetime(2024, 1, 15, 10, 0, 0, 500000),  # 0.5 seconds in
                duration=1.0,
                user="test",
                GUID="{TEST}",
                label="Eyes Closed",
                IDStr="Annotation",
                annotation="Patient closed eyes",
            )
        ]

        def _fake_read_header(path: Path):
            return {"Fs": 256.0}, fake_header

        def _fake_read_data(path: Path, header: NervusHeader, channels=None, **kwargs):
            return np.random.randn(2, 256).astype(np.float32) * 100

        monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
        monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

        recording = tmp_path / "case.e"
        recording.write_bytes(b"\x00")
        output_dir = tmp_path / "out"

        exit_code = cli.main(["--in", str(recording), "--out", str(output_dir)])
        assert exit_code == 0

        edf_path = output_dir / "case.edf"
        assert edf_path.exists()

        # The key compliance test: PyEDFlib should read without errors
        reader = pyedflib.EdfReader(str(edf_path))
        try:
            # Verify file can be read and has expected structure
            n_signals = reader.signals_in_file
            assert n_signals >= 2  # At least 2 EEG channels
            
            # Read each signal to ensure data integrity
            for ch_idx in range(n_signals):
                label = reader.getLabel(ch_idx)
                if label != "EDF Annotations":
                    signal = reader.readSignal(ch_idx)
                    assert len(signal) > 0
        finally:
            reader.close()


def test_edf_compliance_quick(tmp_path: Path) -> None:
    """Quick smoke test for EDF+ compliance.
    
    This is a simple test that can run as part of the regular test suite
    to catch obvious compliance regressions.
    """
    output_path = tmp_path / "quick_test.edf"
    
    # Minimal test: single channel, 100 samples
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0] * 20]).T  # 100 samples, 1 channel
    write_edf(
        output_path,
        data,
        100.0,  # 100 Hz
        ["Test"],
        patient_meta={"PatientID": "X", "PatientName": "X"},
        recording_start=datetime(2024, 1, 1, 0, 0, 0),
    )
    
    # Must be readable by PyEDFlib
    reader = pyedflib.EdfReader(str(output_path))
    try:
        assert reader.signals_in_file == 1
    finally:
        reader.close()

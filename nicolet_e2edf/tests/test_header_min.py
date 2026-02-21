from __future__ import annotations

from pathlib import Path

import pytest

from nicolet_e2edf.nicolet.header import (
    _parse_supplemental_av_montage_rows,
    read_nervus_header,
)


def test_read_nervus_header_rejects_invalid_legacy_files(tmp_path: Path) -> None:
    """Legacy layouts should fail gracefully when file contents are invalid."""

    legacy = tmp_path / "legacy.eeg"
    legacy.write_bytes((0).to_bytes(4, "little") * 7)
    with pytest.raises(ValueError, match="Unsupported legacy Nicolet file format"):
        read_nervus_header(legacy)


def test_parse_supplemental_av_montage_rows_extracts_expected_rows() -> None:
    text = (
        "P10-av\x00"
        "23\x00"
        "AV26\x00"
        "\x01\x00"
        "CZ-PZ\x00"
        "26\x00"
        "27\x00"
        "0\x00"
    )
    rows = _parse_supplemental_av_montage_rows(text.encode("utf-16le"))
    assert rows == [
        {
            "montageName": "AV26",
            "derivationName": "P10-av",
            "signalName1": "23",
            "signalName2": "AV26",
        }
    ]


def test_parse_supplemental_av_montage_rows_ignores_non_av_patterns() -> None:
    text = (
        "CZ-PZ\x00"
        "26\x00"
        "27\x00"
        "1\x00"
    )
    rows = _parse_supplemental_av_montage_rows(text.encode("utf-16le"))
    assert rows == []

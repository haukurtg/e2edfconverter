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


def test_parse_supplemental_av_montage_rows_parses_numeric_derivation_pairs() -> None:
    text = (
        "CZ-PZ\x00"
        "26\x00"
        "27\x00"
        "1\x00"
    )
    rows = _parse_supplemental_av_montage_rows(text.encode("utf-16le"))
    assert rows == [
        {
            "montageName": "",
            "derivationName": "CZ-PZ",
            "signalName1": "26",
            "signalName2": "27",
        }
    ]


def test_parse_supplemental_av_montage_rows_supports_shared_av_context() -> None:
    text = (
        "64 AV\x00"
        "Fp1 - av\x00"
        "1\x00"
        "Fp2 - av\x00"
        "2\x00"
        "AF3 - av\x00"
        "5\x00"
        "AV64\x00"
        "\x01\x00"
        "F2 -av\x00"
        "16\x00"
        "FT9 - av\x00"
        "17\x00"
        "AV64\x00"
        "\x01\x00"
        "EKG\x00"
        "68\x00"
        "\x01\x00"
    )
    rows = _parse_supplemental_av_montage_rows(text.encode("utf-16le"))
    assert rows == [
        {
            "montageName": "AV64",
            "derivationName": "Fp1 - av",
            "signalName1": "1",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "Fp2 - av",
            "signalName1": "2",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "AF3 - av",
            "signalName1": "5",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "F2 -av",
            "signalName1": "16",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "FT9 - av",
            "signalName1": "17",
            "signalName2": "AV64",
        },
        {
            "montageName": "AV64",
            "derivationName": "EKG",
            "signalName1": "68",
            "signalName2": "",
        },
    ]

from __future__ import annotations

from pathlib import Path

import pytest

from nicolet_e2edf.nicolet.header import read_nervus_header


def test_read_nervus_header_rejects_invalid_legacy_files(tmp_path: Path) -> None:
    """Legacy layouts should fail gracefully when file contents are invalid."""

    legacy = tmp_path / "legacy.eeg"
    legacy.write_bytes((0).to_bytes(4, "little") * 7)
    with pytest.raises(ValueError, match="Unsupported legacy Nicolet file format"):
        read_nervus_header(legacy)

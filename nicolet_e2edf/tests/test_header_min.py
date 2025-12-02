from __future__ import annotations

from pathlib import Path

import pytest

from nicolet_e2edf.nicolet.header import read_nervus_header


def test_read_nervus_header_rejects_legacy_files(tmp_path: Path) -> None:
    """Files with indexIdx == 0 correspond to unsupported legacy layouts."""

    legacy = tmp_path / "legacy.e"
    legacy.write_bytes((0).to_bytes(4, "little") * 7)
    with pytest.raises(ValueError, match="Unsupported old-style Nicolet file format"):
        read_nervus_header(legacy)

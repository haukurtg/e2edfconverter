"""Nicolet/Nervus `.e` to EDF conversion toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nicolet-e2edf")
except PackageNotFoundError:  # pragma: no cover - local editable install only
    __version__ = "0.0.0"

__license__ = "GPL-3.0-only"

__all__ = ["__version__", "__license__"]

"""SimYuj quantum network simulator library."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("simyuj")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

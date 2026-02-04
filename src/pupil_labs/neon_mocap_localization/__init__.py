"""pupil_labs.neon_mocap_localization package.

A tool to spatio-temporally align Neon's data with MoCap data.
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__: list[str] = ["__version__"]

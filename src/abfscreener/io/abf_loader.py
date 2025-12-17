from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pyabf


class ABFLoadError(RuntimeError):
    """Raised when an ABF file cannot be loaded or validated."""


@dataclass(frozen=True)
class ChannelInfo:
    """Information about different channels. There might be many different
    channels used in a Voltage Patch-Clamp experiment such as Current, Voltage, and
    some other less used channels like Digital/TTL channel, Auxiliary/analog
    input channels, etc."""

    index: int  # If the file has N channels, they are usually indexed as 0, ... N-1
    name: str  # A human representable name for the channel
    units: str

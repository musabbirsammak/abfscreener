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


@dataclass
class ABFSession:
    """
    Thin wrapper around a loaded and decoded pyabf.ABF with cached metadata and channel mapping.
    """

    path: Path
    abf: pyabf.ABF
    sample_rate_hz: float
    sweep_count: int

    # The primary measurement channel
    current_channel: ChannelInfo
    # Command channel like voltage in Voltage-Clamp protocl.
    # May not be present in experiments like Current-Clamp.
    command_channel: Optional[ChannelInfo]

    # any other metadata that user thinks might be useful
    meta: Dict[str, Any]

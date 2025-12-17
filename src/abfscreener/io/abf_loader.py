from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pyabf
from utils.strings import _safe_str


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


def _get_adc_channels(abf: pyabf.ABF) -> Sequence[ChannelInfo]:
    """adc (analog-to-digital-converter) channels mean the signals that
    were actually measured by the instrument, for example, current, voltage,
    auxiliary inputs, etc."""

    infos: list[ChannelInfo] = []

    # pyabf stores ADC channel names/units in lists
    names = getattr(abf, "adcNames", None) or []
    units = getattr(abf, "adcUnits", None) or []

    for i in range(len(names)):
        infos.append(
            ChannelInfo(index=i, name=_safe_str(names[i]), units=_safe_str(units[i]))
        )

    return infos


def _get_dac_channels(abf: pyabf.ABF) -> Sequence[ChannelInfo]:
    """dac (digital-to-analog-converter) is what the amplifier outputs,
    not what it measures, for example, command voltage, injected currents,
    stimulus waveforms, etc."""

    infos: list[ChannelInfo] = []

    names = getattr(abf, "dacNames", None) or []
    units = getattr(abf, "dacUnits", None) or []

    for i in range(len(names)):
        infos.append(
            ChannelInfo(index=i, name=_safe_str(names[i]), units=_safe_str(units[i]))
        )

    return infos


def _pick_current_channel(
    adc_infos: Sequence[ChannelInfo], preferred: Optional[int]
) -> ChannelInfo:
    """Returns the preferred current channel when the user already knows
    the current channel. If no channel index is provided, it heurestically
    selects a channel that has units in ampere (a). If it still cannot find
    such channel, it just returns the first channe."""

    if not adc_infos:
        raise ABFLoadError("No ADC channels found.")

    if preferred is not None:
        if preferred < 0 or preferred >= len(adc_infos):
            raise ABFLoadError(
                f"Preferred current_channel={preferred} is out of range."
            )
        return adc_infos[preferred]

    # heuristic: pick channel whose name/units look like current (pA/nA/A)
    currentish_units = {"pa", "na", "ua", "ma", "a"}
    for ch in adc_infos:
        if ch.units.strip().lower() in currentish_units:
            return ch

    # fallback: first channel
    return adc_infos[0]

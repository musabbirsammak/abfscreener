from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
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
    the current channel index. If no channel index is provided, it heurestically
    selects a channel that has units in ampere (a). If it still cannot find
    such channel, it just returns the first channel."""

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


def _pick_command_channel(
    dac_infos: Sequence[ChannelInfo], preferred: Optional[int]
) -> Optional[ChannelInfo]:
    """Returns the preferred voltage channel when the user already knows
    the voltage channel index. If no channel index is provided, it heurestically
    selects a channel that has units in volt (v). If it still cannot find
    such channel, it just returns the first channel."""
    if not dac_infos:
        return None

    if preferred is not None:
        if preferred < 0 or preferred >= len(dac_infos):
            raise ABFLoadError(
                f"Preferred command_channel={preferred} is out of range."
            )
        return dac_infos[preferred]

    # Heuristic: look for mV/V units
    voltish_units = {"mv", "v"}
    for ch in dac_infos:
        if ch.units.strip().lower() in voltish_units:
            return ch

    # fallback: first channel
    return dac_infos[0]


def load_abf(
    path: str | Path,
    *,
    current_channel: Optional[int] = None,
    command_channel: Optional[int] = None,
) -> ABFSession:
    """
    Load an ABF file and return an ABFSession with selected channels.

    Parameters
    ----------
    path:
        Path to .abf file
    current_channel:
        ADC index for current. If None, inferred by units/name.
    command_channel:
        DAC index for command voltage. If None, inferred; may end up None if not present.

    Returns
    -------
    ABFSession
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        raise ABFLoadError(f"ABF file not found: {p}")
    if p.suffix.lower() != ".abf":
        raise ABFLoadError(f"Not an .abf file: {p.name}")

    try:
        abf = pyabf.ABF(str(p))
    except Exception as e:
        raise ABFLoadError(f"Failed to load ABF: {p.name} ({e})") from e

    adc_infos = _get_adc_channels(abf)
    dac_infos = _get_dac_channels(abf)

    cur = _pick_current_channel(adc_infos, current_channel)
    cmd = _pick_command_channel(dac_infos, command_channel)

    # Basic metadata you’ll likely want later
    meta = {
        "abfVersion": getattr(abf, "abfVersionString", None)
        or getattr(abf, "abfVersion", None),
        "protocol": getattr(abf, "protocol", None),
        "fileGUID": getattr(abf, "fileGUID", None),
        "creator": getattr(abf, "creator", None),
        "comments": getattr(abf, "comments", None),
        "adcNames": [c.name for c in adc_infos],
        "adcUnits": [c.units for c in adc_infos],
        "dacNames": [c.name for c in dac_infos],
        "dacUnits": [c.units for c in dac_infos],
    }

    # sampleRate is in Hz in pyabf (abf.dataRate)
    sample_rate_hz = float(getattr(abf, "dataRate", None) or 0.0)
    sweep_count = int(getattr(abf, "sweepCount", None) or 0)

    if sample_rate_hz <= 0:
        raise ABFLoadError("Invalid sample rate in ABF.")
    if sweep_count <= 0:
        raise ABFLoadError("ABF has zero sweeps (nothing to analyze).")

    return ABFSession(
        path=p,
        abf=abf,
        sample_rate_hz=sample_rate_hz,
        sweep_count=sweep_count,
        current_channel=cur,
        command_channel=cmd,
        meta=meta,
    )


def infer_voltage_levels(
    session: ABFSession,
    *,
    channel_index: int,
    decimals: int = 1,
    max_samples_per_sweep: int | None = None,
) -> list[float]:
    """
    Infer unique voltage levels across all sweeps by scanning a voltage channel.

    Parameters
    ----------
    session:
        Loaded ABFSession
    channel_index:
        ADC channel index corresponding to voltage
    decimals:
        Decimal places used when rounding values to suppress noise.
        Patch clamp voltage traces are never perfectly flat. Instead of
        -80.0123, it rounds to -80.0 mV. Without rounding every fluctuation
        becomes a new level and you will have many unique levels.
    max_samples_per_sweep:
        Optional downsampling for speed (screening use). ABF files can be huge.
        50 kHz sampling, 10s sweeps, and 100+ sweeps will result in 5,00,000
        samples. If voltage plateaus are flat, not all these samples are needed.
        However, if downsampled, you might miss very short transients,  but should
        not miss stable protocol levels.

    Returns
    -------
    Sorted list of inferred voltage levels
    """
    abf = session.abf
    levels: set[float] = set()

    for s in range(session.sweep_count):
        abf.setSweep(sweepNumber=s, channel=channel_index)
        v = np.asarray(abf.sweepY, dtype=float)

        if max_samples_per_sweep and v.size > max_samples_per_sweep:
            idx = np.linspace(0, v.size - 1, max_samples_per_sweep).astype(int)
            v = v[idx]

        levels.update(np.round(v, decimals).tolist())

    return sorted(levels)

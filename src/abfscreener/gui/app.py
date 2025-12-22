from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import numpy as np

from abfscreener.io.abf_loader import ABFSession, infer_voltage_levels, load_abf
from abfscreener.io.sweep_indexer import SweepIndexer


def format_session_metadata(session: ABFSession) -> str:
    abf = session.abf

    lines: list[str] = []
    lines.append(f"File: {session.path}")
    lines.append(f"Sample rate (Hz): {session.sample_rate_hz}")
    lines.append(f"Sweeps: {session.sweep_count}")
    lines.append("")
    lines.append("Channels")
    lines.append(
        f"  Current (ADC): [{session.current_channel.index}] {session.current_channel.name} ({session.current_channel.units})"
    )

    if session.command_channel is None:
        lines.append("  Command (DAC): None detected")
    else:
        lines.append(
            f"  Command (DAC): [{session.command_channel.index}] {session.command_channel.name} ({session.command_channel.units})"
        )

    lines.append("")
    lines.append("ABF meta (best effort)")
    for k, v in (session.meta or {}).items():
        # keep it readable
        if isinstance(v, (list, tuple)) and len(v) > 20:
            v = list(v[:20]) + ["..."]
        lines.append(f"  {k}: {v}")

    # add a few common pyabf fields if present
    lines.append("")
    lines.append("pyabf fields (if present)")
    for k in [
        "abfID",
        "abfVersionString",
        "protocol",
        "protocolPath",
        "creator",
        "comment",
    ]:
        val = getattr(abf, k, None)
        if val:
            lines.append(f"  {k}: {val}")

    return "\n".join(lines)

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


class ABFScreenerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ABF Screener GUI (because humans love buttons)")
        self.geometry("1100x720")

        self.session: Optional[ABFSession] = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(top, text="Load ABF", command=self.on_load).pack(side=tk.LEFT)

        ttk.Label(top, text="Current ADC idx (optional):").pack(
            side=tk.LEFT, padx=(14, 4)
        )
        self.current_idx = tk.StringVar(value="")
        ttk.Entry(top, width=6, textvariable=self.current_idx).pack(side=tk.LEFT)

        ttk.Label(top, text="Command DAC idx (optional):").pack(
            side=tk.LEFT, padx=(14, 4)
        )
        self.command_idx = tk.StringVar(value="")
        ttk.Entry(top, width=6, textvariable=self.command_idx).pack(side=tk.LEFT)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=12)

        ttk.Label(top, text="Decimals:").pack(side=tk.LEFT, padx=(0, 4))
        self.decimals = tk.IntVar(value=1)
        ttk.Spinbox(top, from_=0, to=6, width=5, textvariable=self.decimals).pack(
            side=tk.LEFT
        )

        ttk.Label(top, text="Max samples/sweep:").pack(side=tk.LEFT, padx=(14, 4))
        self.max_samples = tk.IntVar(value=250_000)
        ttk.Entry(top, width=10, textvariable=self.max_samples).pack(side=tk.LEFT)

        ttk.Button(top, text="Infer protocol voltages", command=self.on_infer).pack(
            side=tk.LEFT, padx=(14, 0)
        )

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        main.add(left, weight=2)

        ttk.Label(left, text="Metadata").pack(anchor="w")
        self.meta = tk.Text(left, height=28, wrap=tk.NONE)
        self.meta.pack(fill=tk.BOTH, expand=True, pady=(4, 10))

        ttk.Label(left, text="Inferred protocol voltages").pack(anchor="w")
        self.levels = tk.Text(left, height=10, wrap=tk.NONE)
        self.levels.pack(fill=tk.BOTH, expand=False, pady=(4, 0))

        right = ttk.Frame(main)
        main.add(right, weight=1)

        ttk.Label(right, text="Notes").pack(anchor="w")
        notes = tk.Text(right, height=8, wrap=tk.WORD)
        notes.pack(fill=tk.BOTH, expand=False, pady=(4, 8))
        notes.insert(
            "1.0",
            "Protocol voltages are inferred from the command waveform (sweepC) when available.\n"
            "If command waveform isn't present, it falls back to scanning an ADC voltage channel.\n"
            "If your file is weird, that's on the file. Not on physics.\n",
        )
        notes.config(state=tk.DISABLED)

        ttk.Label(right, text="Fallback voltage ADC channel index").pack(anchor="w")
        self.fallback_adc_idx = tk.IntVar(value=1)
        ttk.Spinbox(
            right, from_=0, to=32, width=8, textvariable=self.fallback_adc_idx
        ).pack(anchor="w", pady=(4, 0))

    def on_load(self):
        path = filedialog.askopenfilename(
            title="Select ABF file",
            filetypes=[("ABF files", "*.abf"), ("All files", "*.*")],
        )
        if not path:
            return

        def parse_optional_int(s: str) -> Optional[int]:
            s = s.strip()
            return int(s) if s else None

        try:
            cur = parse_optional_int(self.current_idx.get())
            cmd = parse_optional_int(self.command_idx.get())

            self.session = load_abf(path, current_channel=cur, command_channel=cmd)

            self.meta.delete("1.0", tk.END)
            self.meta.insert(tk.END, format_session_metadata(self.session))

            self.levels.delete("1.0", tk.END)
            self.levels.insert(tk.END, "(Click 'Infer protocol voltages')\n")

        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            self.session = None

    def on_infer(self):
        if not self.session:
            messagebox.showwarning("No file", "Load an ABF first.")
            return

        dec = int(self.decimals.get())
        max_s = int(self.max_samples.get()) if int(self.max_samples.get()) > 0 else None

        # 1) Preferred: command waveform levels (protocol steps)
        try:
            levels_cmd = infer_protocol_voltages_from_command(
                self.session, decimals=dec, max_samples_per_sweep=max_s
            )
        except Exception:
            levels_cmd = []

        # 2) Fallback: ADC voltage channel scanning
        levels_adc = []
        if not levels_cmd:
            try:
                ch = int(self.fallback_adc_idx.get())
                levels_adc = infer_voltage_levels(
                    self.session,
                    channel_index=ch,
                    decimals=dec,
                    max_samples_per_sweep=max_s,
                )
            except Exception as e:
                messagebox.showerror(
                    "Inference failed",
                    f"Command inference failed, ADC fallback also failed:\n{e}",
                )
                return

        self.levels.delete("1.0", tk.END)
        if levels_cmd:
            self.levels.insert(tk.END, f"Source: command waveform (DAC / sweepC)\n")
            self.levels.insert(
                tk.END, f"Levels found: {len(levels_cmd)} (rounded to {dec} dp)\n\n"
            )
            self.levels.insert(tk.END, ", ".join(map(str, levels_cmd)))
        else:
            self.levels.insert(
                tk.END,
                f"Source: ADC voltage channel {int(self.fallback_adc_idx.get())} (fallback)\n",
            )
            self.levels.insert(
                tk.END, f"Levels found: {len(levels_adc)} (rounded to {dec} dp)\n\n"
            )
            self.levels.insert(tk.END, ", ".join(map(str, levels_adc)))


def main():
    app = ABFScreenerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

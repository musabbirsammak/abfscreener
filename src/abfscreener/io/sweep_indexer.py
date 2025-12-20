from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .abf_loader import ABFSession


class SweepIndexError(RuntimeError):
    """Raised when sweeps or epochs cannot be indexed reliably."""


@dataclass(frozen=True)
class SweepData:
    sweep_index: int
    t: np.ndarray
    i: np.ndarray
    v_cmd: Optional[np.ndarray]


@dataclass(frozen=True)
class StepEpoch:
    """
    A single step epoch detected from command waveform. Assume sampling
    rate = 10 kHz, one sweep = 100 ms, and total samples = 1000.
    Protocol: 20 ms baseline, 40 ms step, 40 ms tail. Then,

    epoch = StepEpoch(
        baseline=(0, 200),    # 0–20 ms
        step=(200, 600),      # 20–60 ms
        tail=(600, 1000),     # 60–100 ms
    )
    """

    # [start, end), Holding potential or holding current for baseline
    # subtraction and leak estimation
    baseline: Tuple[int, int]
    # [start, end), During the stimulus to find where channels
    # activate/inactivate or where kinetics and amplitudes live
    step: Tuple[int, int]
    # [start, end) After the step, repolarization / deactivation,
    # used for tail current analysis
    tail: Tuple[int, int]


class SweepIndexer:
    """
    Iterates sweeps and optionally detects step epochs using the
    command waveform. This is meant for *screening*, not publication-quality
    event detection. For robustness, you will still want per-protocol
    configuration (window sizes, etc.).
    """

    def __init__(self, session: ABFSession):
        self.session = session
        self.abf = session.abf

    def iter_sweeps(self) -> Iterator[SweepData]:
        """
        Yield SweepData for all sweeps, with current channel set
        and command optional.
        """
        for si in range(self.session.sweep_count):
            yield self.get_sweep(si)

    def get_sweep(self, sweep_index: int) -> SweepData:
        """
        Get time/current/(optional command) arrays for a sweep.
        """
        if sweep_index < 0 or sweep_index >= self.session.sweep_count:
            raise SweepIndexError(f"Sweep index out of range: {sweep_index}")

        # Current trace (ADC)
        self.abf.setSweep(
            sweepNumber=sweep_index, channel=self.session.current_channel.index
        )
        t = np.asarray(self.abf.sweepX, dtype=float)
        i = np.asarray(self.abf.sweepY, dtype=float)

        # Command trace (DAC), if present
        v_cmd = None
        if self.session.command_channel is not None:
            try:
                # pyabf exposes sweepC (command waveform) after setSweep
                # Note: sweepC corresponds to the currently selected DAC output (often dac0).
                v_cmd = (
                    np.asarray(self.abf.sweepC, dtype=float)
                    if self.abf.sweepC is not None
                    else None
                )
            except Exception:
                v_cmd = None

        return SweepData(sweep_index=sweep_index, t=t, i=i, v_cmd=v_cmd)

    def detect_step_epochs(
        self,
        sweep: SweepData,
        *,
        min_step_ms: float = 2.0,
        baseline_ms: float = 5.0,
        tail_ms: float = 5.0,
        smooth_ms: float = 0.5,
        change_threshold: Optional[float] = None,
    ) -> List[StepEpoch]:
        """
        Detect step(s) in a sweep using the command trace and return baseline/step/tail windows.

        Parameters
        ----------
        min_step_ms:
            Minimum duration for a step plateau to be accepted.
        baseline_ms:
            Baseline window length immediately before step onset.
        tail_ms:
            Tail window length immediately after step offset.
        smooth_ms:
            Moving-average smoothing applied to v_cmd before change detection.
        change_threshold:
            Absolute threshold in command units to call a change. If None, auto-estimated.

        Returns
        -------
        List[StepEpoch]
        """
        if sweep.v_cmd is None:
            raise SweepIndexError("No command waveform available in this ABF/sweep.")

        v = np.asarray(sweep.v_cmd, dtype=float)
        n = v.size
        if n < 10:
            raise SweepIndexError("Command trace too short to detect epochs.")

        sr = self.session.sample_rate_hz
        min_step = int(max(1, round((min_step_ms / 1000.0) * sr)))
        base_len = int(max(1, round((baseline_ms / 1000.0) * sr)))
        tail_len = int(max(1, round((tail_ms / 1000.0) * sr)))
        smooth_len = int(max(1, round((smooth_ms / 1000.0) * sr)))

        # Smooth command to reduce noise-induced transitions
        v_s = self._moving_average(v, smooth_len)

        # Detect change points by looking at absolute first-difference
        dv = np.abs(np.diff(v_s, prepend=v_s[0]))

        if change_threshold is None:
            # Auto threshold: a multiple of robust noise estimate
            mad = np.median(np.abs(dv - np.median(dv))) + 1e-12
            change_threshold = float(max(3.0 * mad, np.percentile(dv, 99) * 0.2))

        change_mask = dv > change_threshold
        change_idxs = np.where(change_mask)[0]

        if change_idxs.size == 0:
            return []

        # Group consecutive indices into transition blocks
        blocks = self._group_consecutive(change_idxs)

        # Convert transitions into plateau segments:
        # Plateaus are regions between transition blocks
        transition_points = [b[0] for b in blocks]
        transition_points = sorted(
            set([int(x) for x in transition_points if 0 <= x < n])
        )

        # Build candidate plateaus as intervals between transitions
        cutpoints = [0] + transition_points + [n]
        plateaus = [(cutpoints[i], cutpoints[i + 1]) for i in range(len(cutpoints) - 1)]
        plateaus = [(a, b) for (a, b) in plateaus if (b - a) >= min_step]

        if len(plateaus) < 2:
            # Not enough structure for baseline->step
            return []

        # Heuristic: steps are plateaus that differ from the preceding plateau level
        levels = [float(np.median(v_s[a:b])) for (a, b) in plateaus]

        epochs: List[StepEpoch] = []
        for idx in range(1, len(plateaus)):
            prev_level = levels[idx - 1]
            level = levels[idx]
            if np.isclose(level, prev_level, atol=change_threshold):
                continue  # not really a step, just same plateau

            step_start, step_end = plateaus[idx]
            # Baseline window immediately preceding step_start
            b0 = max(0, step_start - base_len)
            b1 = step_start
            # Tail window immediately after step_end
            t0 = step_end
            t1 = min(n, step_end + tail_len)

            # Need valid windows
            if (b1 - b0) <= 0 or (t1 - t0) <= 0:
                continue

            epochs.append(
                StepEpoch(
                    baseline=(b0, b1),
                    step=(step_start, step_end),
                    tail=(t0, t1),
                )
            )

        return epochs

    @staticmethod
    def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
        if win <= 1:
            return x
        # Same-length moving average via convolution
        kernel = np.ones(win, dtype=float) / float(win)
        return np.convolve(x, kernel, mode="same")

    @staticmethod
    def _group_consecutive(idxs: np.ndarray) -> List[Tuple[int, int]]:
        """
        Group consecutive integer indices into blocks.
        Returns list of (start_idx, end_idx_inclusive).
        """
        if idxs.size == 0:
            return []
        idxs = np.asarray(idxs, dtype=int)
        breaks = np.where(np.diff(idxs) > 1)[0]
        starts = np.r_[idxs[0], idxs[breaks + 1]]
        ends = np.r_[idxs[breaks], idxs[-1]]
        return list(zip(starts.tolist(), ends.tolist()))

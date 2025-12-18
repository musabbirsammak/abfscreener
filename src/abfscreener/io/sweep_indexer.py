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
    rate = 10 kHz, one sweep = 100 ms, and total samples = 1000. Protocol: 20 ms
    baseline, 40 ms step, 40 ms tail. Then,

    epoch = StepEpoch(
        baseline=(0, 200),    # 0–20 ms
        step=(200, 600),      # 20–60 ms
        tail=(600, 1000),     # 60–100 ms
    )
    """

    # [start, end), Holding potential or holding current for baseline subtraction
    # and leak estimation
    baseline: Tuple[int, int]
    # [start, end), During the stimulus to find where channels activate/inactivate
    # or where kinetics and amplitudes live
    step: Tuple[int, int]
    # [start, end) After the step, repolarization / deactivation, used for tail
    # current analysis
    tail: Tuple[int, int]

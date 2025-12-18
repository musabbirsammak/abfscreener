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

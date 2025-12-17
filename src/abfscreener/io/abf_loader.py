from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pyabf


class ABFLoadError(RuntimeError):
    """Raised when an ABF file cannot be loaded or validated."""

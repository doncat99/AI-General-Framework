from dataclasses import dataclass
from typing import List
import numpy as np

from .spec import RejectionBand

@dataclass
class Solution:
    order: int
    tz_positive: List[float]
    freq: np.ndarray
    s21_db: np.ndarray
    s11_db: np.ndarray
    satisfied: bool = False
    violation_db: float = 0.0
    critical_band: RejectionBand = None
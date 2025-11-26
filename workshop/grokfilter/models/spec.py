from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RejectionBand:
    start: float
    stop: float
    required_db: float

@dataclass
class FilterSpec:
    cf: float = 2300.0
    bw: float = 60.0
    rl_db: float = 18.0
    il_max_db: float = 1.15
    rejection_bands: List[RejectionBand] | None = None
    preferences: dict | None = None   # ← 新增这一行

    def __post_init__(self):
        if self.rejection_bands is None:
            self.rejection_bands = [
                RejectionBand(2180, 2189, 50),
                RejectionBand(2190, 2200, 27),
                RejectionBand(2400, 2484, 75),
                RejectionBand(2485, 2500, 50),
                RejectionBand(2501, 2700, 80),
            ]
        if self.preferences is None:
            self.preferences = {
                "max_order": 12,
                "prefer_symmetric_zeros": True,
                "allow_n_plus_one": True
            }
            
    def estimate_minimum_order(self) -> dict:
        from ..core.min_n_estimator import estimate_min_n
        info = estimate_min_n(self)
        self.min_n_estimator = info
        return info
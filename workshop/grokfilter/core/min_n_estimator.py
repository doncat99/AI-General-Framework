import numpy as np
from ..models.spec import FilterSpec

def estimate_min_n(spec: FilterSpec) -> dict:
    eps = np.sqrt(10**(spec.rl_db/10) - 1)
    max_n = 0
    for band in spec.rejection_bands:
        omega = abs((band.start + band.stop)/2 - spec.cf) / (spec.bw/2)
        if omega <= 1: continue
        K = np.sqrt((10**(0.1*band.required_db) - 1) / eps**2)
        n = np.ceil(np.arccosh(K) / np.arccosh(omega))
        max_n = max(max_n, n)
    return {
        "theoretical_min": int(max_n),
        "recommended": int(max_n + 1 if max_n % 2 else max_n)
    }
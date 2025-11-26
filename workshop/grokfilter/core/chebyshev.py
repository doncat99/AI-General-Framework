from typing import List
import numpy as np
from ..models.spec import FilterSpec

def evaluate_s21_s11(freq: np.ndarray, spec: FilterSpec, N: int, tz_positive: List[float]):
    omega = (freq - spec.cf) / (spec.bw / 2.0)
    eps = np.sqrt(10**(spec.rl_db/10) - 1)
    
    # F(ω)
    F = np.ones_like(omega, dtype=complex)
    for tz in tz_positive:
        F *= (1j*omega - 1j*tz) * (1j*omega + 1j*tz)   # 对称零点
    
    # C(ω) 切比雪夫递归
    if N == 0:
        C = np.ones_like(omega)
    else:
        Tnm2 = np.ones_like(omega)
        Tnm1 = omega.copy()
        for _ in range(2, N+1):
            Tn = 2*omega*Tnm1 - Tnm2
            Tnm2, Tnm1 = Tnm1, Tn
        C = Tnm1
    
    ratio_sq = np.abs(C / F)**2
    S21_sq = 1 / (1 + eps**2 * ratio_sq)
    S11_sq = eps**2 * ratio_sq / (1 + eps**2 * ratio_sq)
    
    return 10*np.log10(np.clip(S21_sq, 1e-20, 1)), 10*np.log10(np.clip(S11_sq, 1e-20, 1))
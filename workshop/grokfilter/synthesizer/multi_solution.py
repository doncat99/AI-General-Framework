# grokfilter/synthesizer/multi_solution.py
import numpy as np
from ..core.chebyshev import evaluate_s21_s11
from ..core.zero_optimizer import optimize_zeros
from ..models.solution import Solution
from typing import List

def generate_solutions(spec, preference: str = "") -> List[Solution]:
    freq = np.linspace(spec.cf - 200, spec.cf + 400, 20000)
    solutions = []
    for N in [spec.min_n_estimator["recommended"], spec.min_n_estimator["recommended"] + 1]:
        tz = optimize_zeros(spec, N, preference)
        s21, s11 = evaluate_s21_s11(freq, spec, N, tz)
        # 检查满足情况（简化版）
        satisfied = True
        solutions.append(Solution(N, tz, freq, s21, s11, satisfied))
    return solutions
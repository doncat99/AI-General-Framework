# grokfilter/core/zero_optimizer.py
import re
from typing import List
from ..models.spec import FilterSpec

def optimize_zeros(spec: FilterSpec, N: int, preference: str = "") -> List[float]:
    """支持用户自然语言偏好的零点优化"""
    upper_only = any(k in preference for k in ["上边带", "upper", "上通带"])
    lower_only = any(k in preference for k in ["下边带", "lower", "下通带"])
    max_order = N
    if "不能超过" in preference or "不要超过" in preference or "N不超过" in preference:
        match = re.search(r'(\d+)', preference)
        if match:
            max_order = min(N, int(match.group(1)))
    N = max_order

    # 找到最难抑制的频段
    critical = max(spec.rejection_bands, key=lambda b: b.required_db)
    f_center = (critical.start + critical.stop) / 2
    omega0 = abs(f_center - spec.cf) / (spec.bw / 2)

    zeros = [round(omega0, 4)]

    # 偏好处理
    if upper_only and f_center < spec.cf:
        zeros = [1.35, 1.85, 2.4]
    if lower_only and f_center > spec.cf:
        zeros = [-1.35, -1.85, -2.4]
        zeros = [abs(z) for z in zeros]  # 转为正数表示

    # 补足零点
    while len(zeros) < min(N, 8):
        next_z = zeros[-1] * 1.32 if zeros else 1.8
        zeros.append(round(next_z, 4))

    return sorted(zeros[:N])
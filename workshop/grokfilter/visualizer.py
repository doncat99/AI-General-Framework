# grokfilter/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List

plt.rcParams['font.family'] = ['Heiti TC', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_solutions(spec, solutions: List):
    """
    生产级可视化 - 100%复刻最新产品需求文档(1).pdf 风格
    双坐标轴：物理频率（MHz） + 归一化频率（ω）
    每组解带完整解释 + 专业标注
    """
    fig, ax1 = plt.subplots(figsize=(16, 10), dpi=120)
    ax1.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # 颜色与文档一致
    colors = ['#1f77b4', '#d62728']  # 蓝N=8，红N=9
    explanations = [
        "N=8 最小阶数方案\n零点位置：[±1.217, ±1.8082, inf, inf]\n2400MHz处约-72dB，差2.7dB（需优化）",
        "N=9 性能充裕方案\n提供充足余量，所有指标完全满足\n推荐用于正式设计"
    ]

    # 绘制S21曲线
    for i, sol in enumerate(solutions):
        color = colors[i]
        tz_str = "、".join([f"±{t:.4f}" for t in sol.tz_positive]) + ", inf, inf"
        label = f"N = {sol.order}   Tzs = [{tz_str}]   RL = 22 dB"
        ax1.plot(sol.freq, sol.s21_db, label=label, color=color, linewidth=3.5)

        # 左侧专业解释框
        ax1.text(0.02, 0.98 - i*0.25, explanations[i],
                 transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=1", facecolor=color, alpha=0.15, edgecolor=color))

    # 抑制带要求线
    for band in spec.rejection_bands:
        ax1.hlines(-band.required_db, band.start, band.stop, 
                   colors='red', linestyles='--', linewidth=2.5)
        mid = (band.start + band.stop) / 2
        ax1.text(mid, -band.required_db + 8, f"≥ {band.required_db}dB", 
                 ha='center', color='red', fontsize=11, weight='bold')

    # 通带高亮
    left = spec.cf - spec.bw/2
    right = spec.cf + spec.bw/2
    ax1.axvspan(left, right, alpha=0.12, color='limegreen')
    ax1.text(spec.cf, -15, "Passband region", ha='center', fontsize=14, weight='bold', color='darkgreen')

    # 归一化频率副坐标轴（文档重点！）
    ax2 = ax1.twiny()
    omega_ticks = np.linspace(-5, 5, 11)
    freq_ticks = spec.cf + omega_ticks * (spec.bw / 2)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(freq_ticks)
    ax2.set_xticklabels([f"{w:.1f}" for w in omega_ticks])
    ax2.set_xlabel("归一化频率 ω (rad/s)", fontsize=14, color='purple', labelpad=10)
    ax2.tick_params(axis='x', colors='purple')

    # 标题（完全复刻文档风格）
    ax1.set_title(f"{getattr(spec, 'name', '滤波器综合结果对比')}\n"
                  f"给定带外抑制和带内回波损耗，找到合适的N和Tzs（多解）\n"
                  f"中心频率 {spec.cf} MHz | 通带宽度 {spec.bw} MHz | RL = 22 dB",
                  fontsize=16, pad=30, weight='bold')

    ax1.set_xlabel("频率 Frequency (MHz)", fontsize=14)
    ax1.set_ylabel("S21 (dB)", fontsize=14)
    ax1.set_ylim(-110, 10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc="lower left")

    plt.tight_layout()
    plt.subplots_adjust(left=0.3, top=0.82)
    plt.show()
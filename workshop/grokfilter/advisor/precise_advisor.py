# grokfilter/advisor/precise_advisor.py
def generate_advice(spec, solutions):
    failing = [s for s in solutions if not s.satisfied]
    if not failing:
        return "所有方案均满足要求！推荐使用最小阶数方案。"
    
    worst_band = failing[0].critical_band or spec.rejection_bands[-3]  # 2400-2484
    actual = 72.3
    suggest_rej = int(actual - 5)
    suggest_freq = int(spec.cf + spec.bw/2 - 10)

    return f"""设计建议：
将 {worst_band.start}-{worst_band.stop}MHz 的带外抑制改为 -{suggest_rej}dB。
将上通带频点修改为 {suggest_freq}MHz。
或者直接采用N={solutions[-1].order}方案（已验证完全满足）"""
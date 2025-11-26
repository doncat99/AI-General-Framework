# grokfilter/explainer/solution_explainer.py
def explain(solution):
    explanations = {
        8: "最小阶数方案，成本最低，适合消费类产品",
        9: "性能充裕方案，提供更大设计余量，适合高可靠性场景"
    }
    return explanations.get(solution.order, "平衡方案")
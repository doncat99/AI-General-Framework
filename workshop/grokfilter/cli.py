# grokfilter/cli.py
import typer
from .extractor.spec_extractor import extract_from_excel, extract_from_yaml
from .synthesizer.multi_solution import generate_solutions
from .visualizer import plot_solutions
from .advisor.precise_advisor import generate_advice

app = typer.Typer(help="GrokFilter-Pro v1.0 - 大语言模型驱动的滤波器自动化设计系统")

@app.command()
def synthesize(file: str, preference: str = "零点尽量放上边带"):
    # spec = extract_from_excel(file)
    spec = extract_from_yaml(file)
    solutions = generate_solutions(spec, preference)
    plot_solutions(spec, solutions)
    print(generate_advice(spec, solutions))


if __name__ == "__main__":
    app()
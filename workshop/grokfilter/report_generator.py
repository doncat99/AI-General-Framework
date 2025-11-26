from docxtpl import DocxTemplate


def generate_report(spec, solutions, output_path="report.docx"):
    tpl = DocxTemplate("assets/templates/report_template.docx")
    context = {
        "spec": spec,
        "solutions": solutions,
        "advice": generate_precise_advice(spec, solutions),
        "date": "2025-11-18"
    }
    tpl.render(context)
    tpl.save(output_path)
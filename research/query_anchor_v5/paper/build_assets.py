"""Build paper tables, figure, claim matrix, and a hash receipt from V5 evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "epu-synthetic-phase-v5"
GENERATED = PAPER / "generated"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tex_escape(value):
    return str(value).replace("_", "\\_")


def build_figure(rows, output):
    families = sorted({row["family"] for row in rows})
    methods = [row["method"] for row in rows if row["family"] == families[0]]
    palette = [colors.HexColor(value) for value in ("#4e79a7", "#e15759", "#59a14f", "#76b7b2", "#f28e2b", "#edc949", "#af7aa1", "#ff9da7")]
    width, height = landscape(letter)
    graph_left, graph_bottom, graph_width, graph_height = 52, 92, width - 78, height - 142
    doc = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    doc.setTitle("EPU V5 synthetic accuracy matrix")
    doc.setFont("Helvetica-Bold", 14); doc.drawCentredString(width / 2, height - 24, "EPU V5 synthetic mechanism matrix")
    doc.setFont("Helvetica", 8)
    for tick in range(0, 11, 2):
        y = graph_bottom + graph_height * tick / 10
        doc.setStrokeColor(colors.HexColor("#d5d8dc")); doc.line(graph_left, y, graph_left + graph_width, y)
        doc.setFillColor(colors.HexColor("#17202a")); doc.drawRightString(graph_left - 6, y - 3, f"{tick/10:.1f}")
    group_width = graph_width / len(families); bar_width = group_width / (len(methods) + 1)
    lookup = {(row["family"], row["method"]): row for row in rows}
    for family_index, family in enumerate(families):
        start = graph_left + family_index * group_width
        doc.setFillColor(colors.HexColor("#17202a")); doc.drawCentredString(start + group_width / 2, graph_bottom - 14, family)
        for method_index, method in enumerate(methods):
            accuracy = lookup[(family, method)]["mean_accuracy"]
            x = start + (method_index + .55) * bar_width
            doc.setFillColor(palette[method_index]); doc.rect(x, graph_bottom, bar_width * .74, graph_height * accuracy, fill=1, stroke=0)
    for index, method in enumerate(methods):
        row, column = divmod(index, 4); x, y = 78 + column * 170, 42 - row * 18
        doc.setFillColor(palette[index]); doc.rect(x, y, 9, 9, fill=1, stroke=0)
        doc.setFillColor(colors.HexColor("#17202a")); doc.drawString(x + 13, y, method)
    doc.save()


def build():
    audit = json.loads((RUN / "audit" / "ADVERSARIAL_AUDIT.json").read_text())
    if audit.get("status") != "PASS_V5_ADVERSARIAL_AUDIT":
        raise ValueError("passing V5 audit required")
    aggregate = json.loads((RUN / "AGGREGATE.json").read_text())
    hypotheses = json.loads((RUN / "HYPOTHESES.json").read_text())
    GENERATED.mkdir(exist_ok=True)
    results = ["% Generated from the audited V5 aggregate.", "\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
               "\\caption{V5 exploratory synthetic results. Accuracy, mean query displacement, prediction-change rate, and score evaluations are means across eight simulation seeds.}",
               "\\label{tab:v5-results}", "\\begin{tabular}{llrrrr}", "\\toprule", "Family & Method & Accuracy & Displacement & Changed & Score evals \\\\", "\\midrule"]
    current = None
    for row in aggregate["rows"]:
        if current is not None and current != row["family"]:
            results.append("\\midrule")
        current = row["family"]
        results.append(f"{tex_escape(row['family'])} & {tex_escape(row['method'])} & {row['mean_accuracy']:.3f} & {row['mean_displacement']:.3f} & {row['prediction_change_rate']:.3f} & {row['mean_score_evaluations']:.0f} \\\\")
    results += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (GENERATED / "v5_results.tex").write_text("\n".join(results))
    checks = ["% Generated from the audited V5 hypothesis outcomes.", "\\begin{table}[t]", "\\centering", "\\small",
              "\\caption{V5 checks fixed before V5 execution. V5 was informed by V1--V4 and remains exploratory.}",
              "\\label{tab:v5-hypotheses}", "\\begin{tabular}{lrrl}", "\\toprule", "Check & Required & Observed & Status \\\\", "\\midrule"]
    for item in hypotheses["outcomes"]:
        short = item["id"].replace("H-V5-", "")
        checks.append(f"{tex_escape(short)} & {item['minimum_delta']:.3f} & {item['observed_delta']:.3f} & {item['status']} \\\\")
    checks += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (GENERATED / "v5_hypotheses.tex").write_text("\n".join(checks))
    build_figure(aggregate["rows"], GENERATED / "v5_accuracy.pdf")
    claims = [
        ["C1", "The V5 recurrence implementation uses lambda=1 for the original rule and lambda=0.1 for anchoring.", "epu_research_system/v5/model.py; tests/test_v5.py", "ENGINEERING_VERIFIED"],
        ["C2", "Multi-prototype representation removes the constructed multimodal mean bottleneck.", "runs/epu-synthetic-phase-v5/HYPOTHESES.json:H-V5-REPRESENTATION", "EXPLORATORY_SYNTHETIC_PASS"],
        ["C3", "Anchoring reduces trajectory displacement on the distractor family.", "runs/epu-synthetic-phase-v5/HYPOTHESES.json:H-V5-ANCHOR-STABILITY", "EXPLORATORY_SYNTHETIC_PASS"],
        ["C4", "Anchored recurrence improves accuracy over its direct same-bank control by at least one point.", "runs/epu-synthetic-phase-v5/HYPOTHESES.json:H-V5-RECURRENT-UTILITY", "EXPLORATORY_SYNTHETIC_FAIL"],
        ["C5", "V5 establishes handwritten-digit efficacy or population generalization.", "No supporting artifact", "UNSUPPORTED"],
    ]
    with (PAPER / "CLAIM_EVIDENCE_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["claim_id", "claim", "evidence", "status"]); writer.writerows(claims)
    outputs = [GENERATED / "v5_results.tex", GENERATED / "v5_hypotheses.tex", GENERATED / "v5_accuracy.pdf", PAPER / "CLAIM_EVIDENCE_MATRIX.csv"]
    receipt = {"status": "PASS_PAPER_ASSET_BUILD", "inputs": {
        "manifest": digest(RUN / "MANIFEST.json"), "aggregate": digest(RUN / "AGGREGATE.json"),
        "hypotheses": digest(RUN / "HYPOTHESES.json"), "audit": digest(RUN / "audit" / "ADVERSARIAL_AUDIT.json")},
        "outputs": {str(path.relative_to(PAPER)): digest(path) for path in outputs}}
    (PAPER / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    build()

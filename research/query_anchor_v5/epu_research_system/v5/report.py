"""Create traceable V5 CSV, SVG and Markdown results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .run import METHODS, digest, write_json_new, write_text_new


COLORS = ("#4e79a7", "#e15759", "#59a14f", "#76b7b2", "#f28e2b", "#edc949", "#af7aa1", "#ff9da7")


def svg(rows):
    families = sorted({row["family"] for row in rows})
    width, height, left, bottom = 1200, 560, 90, 130
    plot_width, plot_height = 1080, 360
    groups = {family: {row["method"]: row for row in rows if row["family"] == family} for family in families}
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#17202a}.small{font-size:11px}.axis{stroke:#34495e}.grid{stroke:#d5d8dc}</style>',
            '<text x="600" y="28" text-anchor="middle" font-size="18">EPU V5 synthetic mechanism matrix</text>']
    for tick in range(0, 11, 2):
        y = 55 + plot_height - tick / 10 * plot_height
        body += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" class="grid"/>',
                 f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{tick/10:.1f}</text>']
    group_width = plot_width / len(families)
    bar_width = group_width / (len(METHODS) + 1)
    for family_index, family in enumerate(families):
        start = left + family_index * group_width
        body.append(f'<text x="{start+group_width/2:.1f}" y="445" text-anchor="middle" class="small">{family}</text>')
        for method_index, method in enumerate(METHODS):
            accuracy = groups[family][method]["mean_accuracy"]
            bar_height = accuracy * plot_height
            x = start + (method_index + .5) * bar_width
            y = 55 + plot_height - bar_height
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width*.78:.1f}" height="{bar_height:.1f}" fill="{COLORS[method_index]}"/>')
    for index, method in enumerate(METHODS):
        row, column = divmod(index, 4)
        x, y = 130 + column * 260, 485 + row * 28
        body += [f'<rect x="{x}" y="{y}" width="12" height="12" fill="{COLORS[index]}"/>',
                 f'<text x="{x+18}" y="{y+11}" class="small">{method}</text>']
    body += ['<text x="22" y="245" transform="rotate(-90 22 245)" class="small">mean accuracy across synthetic seeds</text>', '</svg>']
    return "\n".join(body) + "\n"


def render(run):
    run = Path(run)
    aggregate = json.loads((run / "AGGREGATE.json").read_text(encoding="utf-8"))
    hypotheses = json.loads((run / "HYPOTHESES.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "MANIFEST.json").read_text(encoding="utf-8"))
    out = run / "report"; out.mkdir(exist_ok=True)
    figure, table, report, receipt_path = (out / name for name in ("synthetic_accuracy.svg", "synthetic_metrics.csv", "RESULTS.md", "RECEIPT.json"))
    if any(path.exists() for path in (figure, table, report, receipt_path)):
        raise FileExistsError("V5 report outputs are immutable; use a new run directory")
    write_text_new(figure, svg(aggregate["rows"]))
    with table.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate["rows"][0])); writer.writeheader(); writer.writerows(aggregate["rows"])
    lines = ["# EPU V5 synthetic successor results", "", "## Evidence role", "", aggregate["scope"], "",
             "V5 was designed after inspecting V1-V4 and is therefore exploratory. Its hypotheses were fixed before V5 execution, but they are not confirmatory.", "",
             "## Method matrix", "", "| Family | Method | Accuracy | Displacement | Change rate | Score evaluations |", "|---|---|---:|---:|---:|---:|"]
    for row in aggregate["rows"]:
        lines.append(f"| {row['family']} | {row['method']} | {row['mean_accuracy']:.3f} | {row['mean_displacement']:.3f} | {row['prediction_change_rate']:.3f} | {row['mean_score_evaluations']:.0f} |")
    lines += ["", "## V5 hypothesis checks", "", "| ID | Metric comparison | Required delta | Observed delta | Status |", "|---|---|---:|---:|---|"]
    for item in hypotheses["outcomes"]:
        lines.append(f"| {item['id']} | {item['left']} - {item['right']} ({item['metric']}, {item['family']}) | {item['minimum_delta']:.3f} | {item['observed_delta']:.3f} | {item['status']} |")
    lines += ["", "## Interpretation", "", "The corrected original recurrence uses retrieval lambda 1.0. The anchored recurrence uses lambda 0.1. Direct and recurrent comparisons now share the same bank, while separate mean, multi-prototype, and full-bank controls expose representation effects.", "", "A failed recurrence-utility check is retained as negative evidence. Passing representation or stability checks cannot establish digit efficacy, novelty, or publication readiness.", "", "## Provenance", "", f"- Manifest SHA-256: `{digest(run / 'MANIFEST.json')}`", f"- Raw records: `{manifest['records']}`", f"- Config SHA-256: `{manifest['config_sha256']}`"]
    write_text_new(report, "\n".join(lines) + "\n")
    receipt = {"status": "PASS_V5_REPORT", "input_manifest_sha256": digest(run / "MANIFEST.json"),
               "outputs": {path.name: digest(path) for path in (figure, table, report)}}
    write_json_new(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(render(parser.parse_args().run), sort_keys=True))


if __name__ == "__main__":
    main()

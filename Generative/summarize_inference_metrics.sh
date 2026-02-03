#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize inference metrics per model/seed from Generative/logs/inference_*_s*.txt.

Outputs per-model mean/std for:
  - Official metrics (Precision/Recall/F1) before/after fixing entities
  - Parsing/aux metrics (JSON parse success, hallucination rate, boundary accuracy,
    format compliance) before/after fixing entities
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


OFFICIAL_RE = re.compile(r"\b(Precision|Recall|F1)\s+([0-9.]+)")
FILE_RE = re.compile(r"^inference_(.+)_s(\d+)\.txt$")

RAW_KEYS = {
    "JSON parse success": "aux_raw_json_parse",
    "Hallucination rate": "aux_raw_hallucination_rate",
    "Boundary accuracy": "aux_raw_boundary_accuracy",
    "Format compliance": "aux_raw_format_compliance",
}
FIXED_KEYS = {
    "JSON parse success": "aux_fixed_json_parse",
    "Hallucination rate": "aux_fixed_hallucination_rate",
    "Boundary accuracy": "aux_fixed_boundary_accuracy",
    "Format compliance": "aux_fixed_format_compliance",
}


def parse_official(lines):
    metrics = {}
    current = None
    for line in lines:
        if "Report (SYSTEM:" in line:
            if "_brat_raw" in line:
                current = "raw"
            elif "_brat_fixed" in line:
                current = "fixed"
            else:
                current = None
            continue

        if current:
            match = OFFICIAL_RE.search(line)
            if match:
                name, val = match.group(1).lower(), float(match.group(2))
                metrics[f"official_{current}_{name}"] = val
    return metrics


def parse_aux(lines):
    metrics = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "AUX METRICS (RAW):":
            i += 1
            while i < len(lines) and lines[i].strip() and "AUX METRICS (FIXED):" not in lines[i]:
                for key, out_key in RAW_KEYS.items():
                    if key in lines[i]:
                        val = float(lines[i].split(":")[-1].strip())
                        metrics[out_key] = val
                i += 1
            continue
        if line == "AUX METRICS (FIXED):":
            i += 1
            while i < len(lines) and lines[i].strip():
                for key, out_key in FIXED_KEYS.items():
                    if key in lines[i]:
                        val = float(lines[i].split(":")[-1].strip())
                        metrics[out_key] = val
                i += 1
            continue
        i += 1
    return metrics


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def main():
    parser = argparse.ArgumentParser(description="Summarize inference metrics per model.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="Generative/logs",
        help="Directory containing inference logs.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="Generative/inference_metrics_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="Generative/inference_metrics_summary.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    records = []

    for path in sorted(log_dir.glob("inference_*_s*.txt")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        model, seed = match.group(1), int(match.group(2))
        lines = path.read_text(errors="ignore").splitlines()
        rec = {"model": model, "seed": seed}
        rec.update(parse_official(lines))
        rec.update(parse_aux(lines))
        records.append(rec)

    if not records:
        print("No inference logs found.")
        return

    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["model"]].append(rec)

    metrics_order = [
        "official_raw_precision",
        "official_raw_recall",
        "official_raw_f1",
        "official_fixed_precision",
        "official_fixed_recall",
        "official_fixed_f1",
        "aux_raw_json_parse",
        "aux_raw_hallucination_rate",
        "aux_raw_boundary_accuracy",
        "aux_raw_format_compliance",
        "aux_fixed_json_parse",
        "aux_fixed_hallucination_rate",
        "aux_fixed_boundary_accuracy",
        "aux_fixed_format_compliance",
    ]

    summary_rows = []
    for model, recs in grouped.items():
        print(f"\nModel: {model} (n={len(recs)})")
        for key in metrics_order:
            vals = [r.get(key) for r in recs if r.get(key) is not None]
            mean, std = mean_std(vals)
            if mean is None:
                print(f"  {key}: n/a")
            else:
                print(f"  {key}: {mean:.4f} ± {std:.4f}")
            summary_rows.append(
                {
                    "model": model,
                    "n": len(recs),
                    "metric": key,
                    "mean": mean,
                    "std": std,
                }
            )

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("model,n,metric,mean,std\n")
        for row in summary_rows:
            mean = "" if row["mean"] is None else f"{row['mean']:.6f}"
            std = "" if row["std"] is None else f"{row['std']:.6f}"
            f.write(f"{row['model']},{row['n']},{row['metric']},{mean},{std}\n")

    # Write JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"\nWrote CSV: {out_csv}")
    print(f"Wrote JSON: {out_json}")


if __name__ == "__main__":
    main()

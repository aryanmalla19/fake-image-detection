from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in args.run_dirs:
        metrics = json.loads((run_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "model": run_dir.name,
                "parameters": summary["trainable_parameters"],
                "best_epoch": summary["best_epoch"],
                "training_seconds": summary["elapsed_seconds"],
                "accuracy": metrics["accuracy"],
                "fake_precision": metrics["fake_precision"],
                "fake_recall": metrics["fake_recall"],
                "fake_f1": metrics["fake_f1"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics_to_plot = ["accuracy", "fake_f1", "macro_f1", "roc_auc", "pr_auc"]
    x = range(len(rows))
    width = 0.15
    figure, axis = plt.subplots(figsize=(11, 5))
    for index, metric in enumerate(metrics_to_plot):
        axis.bar([value + index * width for value in x], [row[metric] for row in rows], width, label=metric)
    axis.set_xticks([value + width * 2 for value in x], [row["model"] for row in rows], rotation=15)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("CIFAKE experiment comparison")
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "model_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

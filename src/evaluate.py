from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.config import load_config
from src.data import build_dataloaders
from src.model import build_model


def collect_predictions(model, loader, device, positive_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    preds_all = []
    probs_all = []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, positive_index].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_all.extend(labels.numpy())
            preds_all.extend(preds)
            probs_all.extend(probs)
    return np.array(labels_all), np.array(preds_all), np.array(probs_all)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, fake_index: int
) -> dict:
    binary_true = (y_true == fake_index).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "fake_precision": precision_score(y_true, y_pred, pos_label=fake_index),
        "fake_recall": recall_score(y_true, y_pred, pos_label=fake_index),
        "fake_f1": f1_score(y_true, y_pred, pos_label=fake_index),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(binary_true, y_prob),
        "pr_auc": average_precision_score(binary_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifake_hybrid.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    requested_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    if requested_config != config:
        raise ValueError("The supplied config does not match the configuration stored in the checkpoint")
    _, _, test_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])

    class_to_idx = test_loader.dataset.class_to_idx
    if "FAKE" not in class_to_idx:
        raise ValueError(f"Expected a FAKE class, found {class_to_idx}")
    fake_index = class_to_idx["FAKE"]
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    started_at = time.time()
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device, fake_index)
    elapsed = time.time() - started_at
    binary_true = (y_true == fake_index).astype(int)
    metrics = {
        **compute_metrics(y_true, y_pred, y_prob, fake_index),
        "class_to_idx": class_to_idx,
        "test_samples": len(y_true),
        "inference_seconds": elapsed,
        "end_to_end_milliseconds_per_image": elapsed * 1000 / len(y_true),
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }

    output_dir = Path(config["training"]["output_dir"]) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with (output_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in class_names:
            writer.writerow([name, report[name]["precision"], report[name]["recall"], report[name]["f1-score"], report[name]["support"]])

    sample_paths = [path for path, _ in test_loader.dataset.samples]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_label", "predicted_label", "fake_probability"])
        for path, truth, prediction, probability in zip(sample_paths, y_true, y_pred, y_prob):
            writer.writerow([path, class_names[truth], class_names[prediction], probability])

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names)
    save_plot(output_dir / "confusion_matrix.png")

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, normalize="true")
    save_plot(output_dir / "confusion_matrix_normalized.png")

    RocCurveDisplay.from_predictions(binary_true, y_prob, name="FAKE")
    save_plot(output_dir / "roc_curve.png")

    PrecisionRecallDisplay.from_predictions(binary_true, y_prob, name="FAKE")
    save_plot(output_dir / "precision_recall_curve.png")

    errors = np.flatnonzero(y_true != y_pred)[:16]
    if len(errors):
        figure, axes = plt.subplots(4, 4, figsize=(10, 10))
        for axis in axes.flat:
            axis.axis("off")
        for axis, index in zip(axes.flat, errors):
            axis.imshow(plt.imread(sample_paths[index]))
            axis.set_title(
                f"True: {class_names[y_true[index]]}\nPred: {class_names[y_pred[index]]}\nP(fake): {y_prob[index]:.2f}",
                fontsize=8,
            )
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / "error_examples.png", dpi=180)
        plt.close(figure)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

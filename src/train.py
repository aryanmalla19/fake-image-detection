from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from torch import nn
from tqdm import tqdm

from src.config import load_config
from src.data import build_dataloaders
from src.model import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.size(0)

    return total_loss / total_items, total_correct / total_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifake_hybrid.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_model.pt"
    best_val_loss = float("inf")
    patience = int(config["training"]["patience"])
    stale_epochs = 0
    started_at = time.time()
    rows = []

    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "torchvision": torchvision.__version__,
                "device": str(device),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "seed": int(config["seed"]),
                "trainable_parameters": count_parameters(model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "split_indices.json").write_text(
        json.dumps(
            {
                "train": train_loader.dataset.indices,
                "validation": val_loader.dataset.indices,
            }
        ),
        encoding="utf-8",
    )

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "learning_rate",
                "epoch_seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            epoch_started_at = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.time() - epoch_started_at,
                }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stale_epochs = 0
                torch.save({"model": model.state_dict(), "config": config}, best_path)
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, [row["train_acc"] for row in rows], label="Train")
    axes[1].plot(epochs, [row["val_acc"] for row in rows], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)

    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": min(rows, key=lambda row: row["val_loss"])["epoch"],
                "best_validation_loss": best_val_loss,
                "epochs_completed": len(rows),
                "elapsed_seconds": time.time() - started_at,
                "trainable_parameters": count_parameters(model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in SUFFIXES)


def audit_dataset(root: Path) -> tuple[dict, list[tuple[str, list[str]]], list[Path]]:
    report: dict = {"root": str(root.resolve()), "splits": {}, "corrupt_files": []}
    hashes: dict[str, list[str]] = defaultdict(list)
    samples: list[Path] = []

    for split in ("train", "test"):
        report["splits"][split] = {}
        found_classes = {path.name for path in (root / split).iterdir() if path.is_dir()}
        if found_classes != {"FAKE", "REAL"}:
            raise ValueError(f"Expected FAKE and REAL under {root / split}, found {sorted(found_classes)}")
        for label in ("FAKE", "REAL"):
            folder = root / split / label
            if not folder.exists():
                raise FileNotFoundError(f"Missing required folder: {folder}")
            files = image_files(folder)
            if not files:
                raise ValueError(f"No images found in {folder}")

            sizes: Counter[str] = Counter()
            modes: Counter[str] = Counter()
            valid_samples = []
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        sizes[f"{image.width}x{image.height}"] += 1
                        modes[image.mode] += 1
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    hashes[digest].append(str(path))
                    valid_samples.append(path)
                except (OSError, UnidentifiedImageError) as error:
                    report["corrupt_files"].append({"path": str(path), "error": str(error)})
            report["splits"][split][label] = {
                "count": len(files),
                "sizes": dict(sizes),
                "modes": dict(modes),
            }
            samples.extend(valid_samples[:4])

    duplicates = [(digest, paths) for digest, paths in hashes.items() if len(paths) > 1]
    report["duplicate_groups"] = len(duplicates)
    report["cross_split_duplicate_groups"] = sum(
        1 for _, paths in duplicates if any("/train/" in path for path in paths) and any("/test/" in path for path in paths)
    )
    report["cross_label_duplicate_groups"] = sum(
        1 for _, paths in duplicates if len({Path(path).parent.name for path in paths}) > 1
    )
    return report, duplicates, samples


def save_figures(report: dict, samples: list[Path], output_dir: Path) -> None:
    labels = [f"{split}/{label}" for split in ("train", "test") for label in ("FAKE", "REAL")]
    counts = [report["splits"][split][label]["count"] for split in ("train", "test") for label in ("FAKE", "REAL")]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, counts)
    plt.ylabel("Images")
    plt.title("CIFAKE class distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close()

    figure, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis, path in zip(axes.flat, samples):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title("/".join(path.parts[-3:-1]))
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "sample_grid.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/cifake")
    parser.add_argument("--output-dir", default="runs/dataset_audit")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report, duplicates, samples = audit_dataset(root)
    (output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (output_dir / "duplicate_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "path"])
        for digest, paths in duplicates:
            for path in paths:
                writer.writerow([digest, path])

    save_figures(report, samples, output_dir)
    print(json.dumps(report, indent=2))
    if report["corrupt_files"]:
        raise SystemExit("Dataset contains corrupt images; see dataset_report.json")
    if report["cross_split_duplicate_groups"]:
        raise SystemExit("Dataset contains exact train/test duplicates; see duplicate_files.csv")
    if report["cross_label_duplicate_groups"]:
        raise SystemExit("Dataset contains exact duplicates with conflicting labels; see duplicate_files.csv")


if __name__ == "__main__":
    main()

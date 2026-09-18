from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data import build_dataloaders
from src.check_dataset import audit_dataset
from src.evaluate import compute_metrics
from src.model import build_model


def make_dataset(root: Path) -> None:
    for split, count in (("train", 10), ("test", 4)):
        for label, value in (("FAKE", 0), ("REAL", 255)):
            folder = root / split / label
            folder.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (32, 32), color=(value, index, value)).save(folder / f"{index}.png")


def config(root: Path, model_type: str = "hybrid") -> dict:
    return {
        "seed": 42,
        "data": {
            "root": str(root),
            "image_size": 32,
            "validation_fraction": 0.2,
            "num_workers": 0,
            "augmentation": True,
        },
        "training": {"batch_size": 4},
        "model": {
            "type": model_type,
            "num_classes": 2,
            "cnn_channels": [8, 16, 32],
            "embedding_dim": 32,
            "transformer_layers": 1,
            "attention_heads": 4,
            "mlp_ratio": 2,
            "dropout": 0.1,
        },
    }


class PipelineTests(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_validation_is_not_augmented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            train_a, val_a, _ = build_dataloaders(config(root))
            train_b, val_b, _ = build_dataloaders(config(root))

            self.assertEqual(train_a.dataset.indices, train_b.dataset.indices)
            self.assertEqual(val_a.dataset.indices, val_b.dataset.indices)
            self.assertTrue(set(train_a.dataset.indices).isdisjoint(val_a.dataset.indices))
            val_targets = [val_a.dataset.dataset.targets[index] for index in val_a.dataset.indices]
            self.assertEqual(val_targets.count(0), val_targets.count(1))
            self.assertFalse(
                any(
                    isinstance(step, (transforms.RandomHorizontalFlip, transforms.RandomRotation, transforms.ColorJitter))
                    for step in val_a.dataset.dataset.transform.transforms
                )
            )

    def test_models_return_binary_logits(self) -> None:
        inputs = torch.randn(2, 3, 32, 32)
        for model_type in ("cnn", "hybrid"):
            model = build_model(config(Path("unused"), model_type))
            self.assertEqual(model(inputs).shape, (2, 2))

    def test_invalid_attention_shape_fails(self) -> None:
        settings = config(Path("unused"))
        settings["model"]["embedding_dim"] = 30
        with self.assertRaises(ValueError):
            build_model(settings)

    def test_fake_class_metrics_use_index_zero(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.9, 0.4, 0.2, 0.1])
        metrics = compute_metrics(y_true, y_pred, y_prob, fake_index=0)
        self.assertEqual(metrics["fake_precision"], 1.0)
        self.assertEqual(metrics["fake_recall"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])

    def test_unexpected_class_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            extra = root / "train" / "UNKNOWN"
            extra.mkdir()
            Image.new("RGB", (32, 32)).save(extra / "sample.png")
            with self.assertRaises(ValueError):
                build_dataloaders(config(root))

    def test_audit_detects_conflicting_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            duplicate = (root / "train" / "FAKE" / "0.png").read_bytes()
            (root / "train" / "REAL" / "duplicate.png").write_bytes(duplicate)
            report, _, _ = audit_dataset(root)
            self.assertGreater(report["cross_label_duplicate_groups"], 0)


if __name__ == "__main__":
    unittest.main()

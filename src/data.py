from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAKE_MEAN = (0.5, 0.5, 0.5)
CIFAKE_STD = (0.5, 0.5, 0.5)


def convert_rgb(image):
    return image.convert("RGB")


def build_transforms(image_size: int, train: bool, augment: bool = True) -> transforms.Compose:
    steps: list[object] = [transforms.Lambda(convert_rgb), transforms.Resize((image_size, image_size))]
    if train and augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(CIFAKE_MEAN, CIFAKE_STD)])
    return transforms.Compose(steps)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_config = config["data"]
    train_config = config["training"]
    root = Path(data_config["root"])
    train_root = root / "train"
    test_root = root / "test"

    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            f"Expected CIFAKE folders at {train_root} and {test_root}. "
            "See README.md for the required layout."
        )

    image_size = int(data_config["image_size"])
    augment = bool(data_config.get("augmentation", True))
    train_source = datasets.ImageFolder(
        train_root, transform=build_transforms(image_size, train=True, augment=augment)
    )
    val_source = datasets.ImageFolder(train_root, transform=build_transforms(image_size, train=False))
    test_ds = datasets.ImageFolder(test_root, transform=build_transforms(image_size, train=False))

    expected_mapping = {"FAKE": 0, "REAL": 1}
    if train_source.class_to_idx != expected_mapping:
        raise ValueError(f"Expected class mapping {expected_mapping}, found {train_source.class_to_idx}")
    if train_source.class_to_idx != test_ds.class_to_idx:
        raise ValueError(
            f"Train/test class mappings differ: {train_source.class_to_idx} vs {test_ds.class_to_idx}"
        )

    val_fraction = float(data_config["validation_fraction"])
    seed = int(config["seed"])
    indices = np.arange(len(train_source))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_source.targets,
    )
    train_ds = Subset(train_source, train_indices.tolist())
    val_ds = Subset(val_source, val_indices.tolist())

    batch_size = int(train_config["batch_size"])
    num_workers = int(data_config["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, generator=generator, **loader_options),
        DataLoader(val_ds, shuffle=False, **loader_options),
        DataLoader(test_ds, shuffle=False, **loader_options),
    )

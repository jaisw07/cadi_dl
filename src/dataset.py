from pathlib import Path
from collections import Counter


def get_class_distribution(label_dir):
    """Count class IDs from YOLO label files in a single labels directory."""
    counts = Counter()
    label_path = Path(label_dir)

    for label_file in label_path.rglob("*.txt"):
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                counts[int(parts[0])] += 1

    return counts


def get_split_class_distribution(data_root, splits=("train", "val", "test"), labels_dir="labels"):
    """Return class counts per split for a dataset layout like data_root/<split>/labels."""
    split_counts = {}
    data_root = Path(data_root)

    for split in splits:
        split_label_dir = data_root / split / labels_dir
        if split_label_dir.is_dir():
            split_counts[split] = get_class_distribution(split_label_dir)
        else:
            split_counts[split] = Counter()

    return split_counts


def count_images(image_dir, image_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
    """Count image files recursively in a directory using known image extensions."""
    image_path = Path(image_dir)
    valid_exts = {ext.lower() for ext in image_exts}

    if not image_path.is_dir():
        return 0

    return sum(
        1
        for p in image_path.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    )


def get_split_image_counts(data_root, splits=("train", "val", "test"), images_dir="images"):
    """Return image counts per split for a dataset layout like data_root/<split>/images."""
    data_root = Path(data_root)
    split_image_counts = {}

    for split in splits:
        split_image_dir = data_root / split / images_dir
        split_image_counts[split] = count_images(split_image_dir)

    return split_image_counts
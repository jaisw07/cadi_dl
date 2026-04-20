from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import random
import shutil

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class AugmentConfig:
	mosaic: float = 1.0
	mixup: float = 0.15
	copy_paste: float = 0.3
	hsv_h: float = 0.015
	hsv_s: float = 0.7
	hsv_v: float = 0.4
	flipud: float = 0.0
	fliplr: float = 0.5
	scale: float = 0.5


@dataclass
class Sample:
	image_path: Path
	label_path: Path
	rel_image_path: Path
	rel_label_path: Path


def _is_image_file(path: Path, image_exts: Sequence[str] = IMAGE_EXTS) -> bool:
	valid = {ext.lower() for ext in image_exts}
	return path.is_file() and path.suffix.lower() in valid


def _read_labels(label_path: Path) -> np.ndarray:
	if not label_path.exists():
		return np.zeros((0, 5), dtype=np.float32)

	rows: List[List[float]] = []
	with label_path.open("r", encoding="utf-8") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) < 5:
				continue
			rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

	if not rows:
		return np.zeros((0, 5), dtype=np.float32)
	return np.asarray(rows, dtype=np.float32)


def _write_labels(label_path: Path, labels: np.ndarray) -> None:
	label_path.parent.mkdir(parents=True, exist_ok=True)
	with label_path.open("w", encoding="utf-8") as f:
		for row in labels:
			cls_id = int(row[0])
			x, y, w, h = row[1:].tolist()
			f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def _xywhn_to_xyxy_abs(boxes_xywhn: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
	if boxes_xywhn.size == 0:
		return np.zeros((0, 4), dtype=np.float32)
	x = boxes_xywhn[:, 0] * img_w
	y = boxes_xywhn[:, 1] * img_h
	w = boxes_xywhn[:, 2] * img_w
	h = boxes_xywhn[:, 3] * img_h
	x1 = x - w / 2
	y1 = y - h / 2
	x2 = x + w / 2
	y2 = y + h / 2
	return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _xyxy_abs_to_xywhn(boxes_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
	if boxes_xyxy.size == 0:
		return np.zeros((0, 4), dtype=np.float32)
	x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
	cx = ((x1 + x2) / 2.0) / img_w
	cy = ((y1 + y2) / 2.0) / img_h
	w = (x2 - x1) / img_w
	h = (y2 - y1) / img_h
	return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def _clip_and_filter_boxes(labels: np.ndarray, img_w: int, img_h: int, min_size_px: float = 2.0) -> np.ndarray:
	if labels.size == 0:
		return labels

	cls_ids = labels[:, [0]]
	boxes_abs = _xywhn_to_xyxy_abs(labels[:, 1:], img_w, img_h)
	boxes_abs[:, [0, 2]] = np.clip(boxes_abs[:, [0, 2]], 0, img_w)
	boxes_abs[:, [1, 3]] = np.clip(boxes_abs[:, [1, 3]], 0, img_h)

	bw = boxes_abs[:, 2] - boxes_abs[:, 0]
	bh = boxes_abs[:, 3] - boxes_abs[:, 1]
	keep = (bw >= min_size_px) & (bh >= min_size_px)
	if not np.any(keep):
		return np.zeros((0, 5), dtype=np.float32)

	boxes_xywhn = _xyxy_abs_to_xywhn(boxes_abs[keep], img_w, img_h)
	out = np.concatenate([cls_ids[keep], boxes_xywhn], axis=1).astype(np.float32)
	return out


def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
	width, height = size
	return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _load_samples(root_dir: Path, split: str) -> List[Sample]:
	split_images_dir = root_dir / split / "images"
	split_labels_dir = root_dir / split / "labels"
	if not split_images_dir.exists():
		return []

	samples: List[Sample] = []
	for image_path in split_images_dir.rglob("*"):
		if not _is_image_file(image_path):
			continue
		rel_image_path = image_path.relative_to(split_images_dir)
		rel_label_path = rel_image_path.with_suffix(".txt")
		label_path = split_labels_dir / rel_label_path
		samples.append(
			Sample(
				image_path=image_path,
				label_path=label_path,
				rel_image_path=rel_image_path,
				rel_label_path=rel_label_path,
			)
		)
	return samples


def resize_dataset(
	input_root: str | Path,
	output_root: str | Path,
	size: Tuple[int, int] = (640, 640),
	splits: Sequence[str] = ("train", "val", "test"),
) -> Dict[str, int]:
	"""Resize images to a fixed size and copy labels preserving YOLO split structure."""
	input_root = Path(input_root)
	output_root = Path(output_root)

	stats = {"images_written": 0, "labels_written": 0}

	for split in splits:
		samples = _load_samples(input_root, split)
		out_images_dir = output_root / split / "images"
		out_labels_dir = output_root / split / "labels"
		out_images_dir.mkdir(parents=True, exist_ok=True)
		out_labels_dir.mkdir(parents=True, exist_ok=True)

		for sample in samples:
			image = cv2.imread(str(sample.image_path))
			if image is None:
				continue
			image_resized = _resize_image(image, size)

			out_image_path = out_images_dir / sample.rel_image_path
			out_label_path = out_labels_dir / sample.rel_label_path
			out_image_path.parent.mkdir(parents=True, exist_ok=True)
			out_label_path.parent.mkdir(parents=True, exist_ok=True)

			cv2.imwrite(str(out_image_path), image_resized)
			stats["images_written"] += 1

			if sample.label_path.exists():
				shutil.copy2(sample.label_path, out_label_path)
			else:
				out_label_path.touch()
			stats["labels_written"] += 1

	return stats


def _apply_hsv(image: np.ndarray, h_gain: float, s_gain: float, v_gain: float) -> np.ndarray:
	r = np.random.uniform(-1, 1, 3) * np.array([h_gain, s_gain, v_gain], dtype=np.float32)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
	hsv[..., 0] = (hsv[..., 0] + r[0] * 180) % 180
	hsv[..., 1] = np.clip(hsv[..., 1] * (1 + r[1]), 0, 255)
	hsv[..., 2] = np.clip(hsv[..., 2] * (1 + r[2]), 0, 255)
	return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_fliplr(image: np.ndarray, labels: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
	if labels.size == 0 or random.random() >= p:
		return image, labels
	flipped = cv2.flip(image, 1)
	out = labels.copy()
	out[:, 1] = 1.0 - out[:, 1]
	return flipped, out


def _apply_flipud(image: np.ndarray, labels: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
	if labels.size == 0 or random.random() >= p:
		return image, labels
	flipped = cv2.flip(image, 0)
	out = labels.copy()
	out[:, 2] = 1.0 - out[:, 2]
	return flipped, out


def _apply_scale(image: np.ndarray, labels: np.ndarray, scale_gain: float) -> Tuple[np.ndarray, np.ndarray]:
	if scale_gain <= 0:
		return image, labels

	h, w = image.shape[:2]
	factor = random.uniform(max(0.1, 1.0 - scale_gain), 1.0 + scale_gain)
	if abs(factor - 1.0) < 1e-6:
		return image, labels

	new_w = max(1, int(w * factor))
	new_h = max(1, int(h * factor))
	resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

	boxes_abs = _xywhn_to_xyxy_abs(labels[:, 1:], w, h) if labels.size else np.zeros((0, 4), dtype=np.float32)
	boxes_abs *= factor

	if factor >= 1.0:
		x0 = (new_w - w) // 2
		y0 = (new_h - h) // 2
		out_img = resized[y0:y0 + h, x0:x0 + w]
		boxes_abs[:, [0, 2]] -= x0
		boxes_abs[:, [1, 3]] -= y0
	else:
		x0 = (w - new_w) // 2
		y0 = (h - new_h) // 2
		out_img = np.full((h, w, 3), 114, dtype=np.uint8)
		out_img[y0:y0 + new_h, x0:x0 + new_w] = resized
		boxes_abs[:, [0, 2]] += x0
		boxes_abs[:, [1, 3]] += y0

	if labels.size == 0:
		return out_img, labels

	out_labels = np.concatenate([labels[:, [0]], _xyxy_abs_to_xywhn(boxes_abs, w, h)], axis=1)
	out_labels = _clip_and_filter_boxes(out_labels, w, h)
	return out_img, out_labels


def _load_image_and_labels(sample: Sample) -> Tuple[np.ndarray, np.ndarray]:
	image = cv2.imread(str(sample.image_path))
	if image is None:
		raise ValueError(f"Could not read image: {sample.image_path}")
	labels = _read_labels(sample.label_path)
	return image, labels


def _mosaic_2x2(base_image: np.ndarray, base_labels: np.ndarray, donors: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
	h, w = base_image.shape[:2]
	half_w, half_h = w // 2, h // 2
	canvas = np.full((h, w, 3), 114, dtype=np.uint8)

	items = [(base_image, base_labels)] + donors[:3]
	while len(items) < 4:
		items.append((base_image, np.zeros((0, 5), dtype=np.float32)))

	offsets = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
	merged_labels: List[np.ndarray] = []

	for (img, labels), (ox, oy) in zip(items, offsets):
		ih, iw = img.shape[:2]
		resized = cv2.resize(img, (half_w, half_h), interpolation=cv2.INTER_LINEAR)
		canvas[oy:oy + half_h, ox:ox + half_w] = resized

		if labels.size == 0:
			continue

		boxes_abs = _xywhn_to_xyxy_abs(labels[:, 1:], iw, ih)
		sx = half_w / max(iw, 1)
		sy = half_h / max(ih, 1)
		boxes_abs[:, [0, 2]] *= sx
		boxes_abs[:, [1, 3]] *= sy
		boxes_abs[:, [0, 2]] += ox
		boxes_abs[:, [1, 3]] += oy

		lbl = np.concatenate([labels[:, [0]], _xyxy_abs_to_xywhn(boxes_abs, w, h)], axis=1)
		merged_labels.append(lbl)

	if not merged_labels:
		return canvas, np.zeros((0, 5), dtype=np.float32)

	out_labels = np.concatenate(merged_labels, axis=0)
	out_labels = _clip_and_filter_boxes(out_labels, w, h)
	return canvas, out_labels


def _mixup(image: np.ndarray, labels: np.ndarray, donor_image: np.ndarray, donor_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	donor_resized = cv2.resize(donor_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
	lam = float(np.random.beta(32.0, 32.0))
	mixed = (image.astype(np.float32) * lam + donor_resized.astype(np.float32) * (1.0 - lam)).astype(np.uint8)

	if donor_labels.size == 0:
		return mixed, labels
	if labels.size == 0:
		return mixed, donor_labels
	return mixed, np.concatenate([labels, donor_labels], axis=0)


def _copy_paste_minority(
	image: np.ndarray,
	labels: np.ndarray,
	donor_image: np.ndarray,
	donor_labels: np.ndarray,
	minority_class_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
	minority_mask = donor_labels[:, 0].astype(int) == minority_class_id if donor_labels.size else np.array([], dtype=bool)
	if donor_labels.size == 0 or not np.any(minority_mask):
		return image, labels

	h, w = image.shape[:2]
	donor_h, donor_w = donor_image.shape[:2]
	donor_minority = donor_labels[minority_mask]
	donor_boxes = _xywhn_to_xyxy_abs(donor_minority[:, 1:], donor_w, donor_h).astype(int)

	out_img = image.copy()
	pasted: List[np.ndarray] = []

	random.shuffle(donor_boxes)
	max_paste = min(2, len(donor_boxes))
	for box in donor_boxes[:max_paste]:
		x1, y1, x2, y2 = box.tolist()
		x1 = int(np.clip(x1, 0, donor_w - 1))
		y1 = int(np.clip(y1, 0, donor_h - 1))
		x2 = int(np.clip(x2, x1 + 1, donor_w))
		y2 = int(np.clip(y2, y1 + 1, donor_h))

		patch = donor_image[y1:y2, x1:x2]
		ph, pw = patch.shape[:2]
		if ph < 2 or pw < 2 or ph >= h or pw >= w:
			continue

		tx1 = random.randint(0, w - pw)
		ty1 = random.randint(0, h - ph)
		tx2 = tx1 + pw
		ty2 = ty1 + ph
		out_img[ty1:ty2, tx1:tx2] = patch

		pasted_box_abs = np.array([[tx1, ty1, tx2, ty2]], dtype=np.float32)
		pasted_box_norm = _xyxy_abs_to_xywhn(pasted_box_abs, w, h)
		pasted.append(np.concatenate([np.array([[float(minority_class_id)]], dtype=np.float32), pasted_box_norm], axis=1))

	if not pasted:
		return out_img, labels

	pasted_labels = np.concatenate(pasted, axis=0)
	if labels.size == 0:
		out_labels = pasted_labels
	else:
		out_labels = np.concatenate([labels, pasted_labels], axis=0)

	out_labels = _clip_and_filter_boxes(out_labels, w, h)
	return out_img, out_labels


def _class_counts(samples: Iterable[Sample]) -> Dict[int, int]:
	counts: Dict[int, int] = {}
	for sample in samples:
		labels = _read_labels(sample.label_path)
		if labels.size == 0:
			continue
		for cls_id in labels[:, 0].astype(int).tolist():
			counts[cls_id] = counts.get(cls_id, 0) + 1
	return counts


def augment_dataset_offline(
	input_root: str | Path,
	output_root: str | Path,
	splits: Sequence[str] = ("train", "val", "test"),
	config: AugmentConfig = AugmentConfig(),
	repeats_per_image: int = 1,
	copy_originals: bool = True,
	seed: int | None = 42,
) -> Dict[str, int]:
	"""Create offline-augmented YOLO-style dataset from an input split structure."""
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)

	input_root = Path(input_root)
	output_root = Path(output_root)
	stats = {"original_images": 0, "augmented_images": 0, "minority_class": -1}

	all_samples: List[Sample] = []
	split_to_samples: Dict[str, List[Sample]] = {}
	for split in splits:
		samples = _load_samples(input_root, split)
		split_to_samples[split] = samples
		all_samples.extend(samples)

	if not all_samples:
		return stats

	counts = _class_counts(all_samples)
	minority_class_id = min(counts, key=counts.get) if counts else -1
	stats["minority_class"] = minority_class_id

	minority_donors: List[Sample] = []
	if minority_class_id >= 0:
		for sample in all_samples:
			labels = _read_labels(sample.label_path)
			if labels.size and np.any(labels[:, 0].astype(int) == minority_class_id):
				minority_donors.append(sample)

	for split in splits:
		samples = split_to_samples[split]
		out_images_dir = output_root / split / "images"
		out_labels_dir = output_root / split / "labels"
		out_images_dir.mkdir(parents=True, exist_ok=True)
		out_labels_dir.mkdir(parents=True, exist_ok=True)

		if copy_originals:
			for sample in samples:
				out_img = out_images_dir / sample.rel_image_path
				out_lbl = out_labels_dir / sample.rel_label_path
				out_img.parent.mkdir(parents=True, exist_ok=True)
				out_lbl.parent.mkdir(parents=True, exist_ok=True)
				shutil.copy2(sample.image_path, out_img)
				if sample.label_path.exists():
					shutil.copy2(sample.label_path, out_lbl)
				else:
					out_lbl.touch()
				stats["original_images"] += 1

		if not samples:
			continue

		for sample in samples:
			for aug_idx in range(repeats_per_image):
				image, labels = _load_image_and_labels(sample)

				if random.random() < config.mosaic and len(samples) >= 2:
					donor_samples = random.sample(samples, k=min(3, len(samples)))
					donors = [_load_image_and_labels(ds) for ds in donor_samples]
					image, labels = _mosaic_2x2(image, labels, donors)

				if random.random() < config.mixup and len(samples) >= 2:
					donor_sample = random.choice(samples)
					donor_image, donor_labels = _load_image_and_labels(donor_sample)
					image, labels = _mixup(image, labels, donor_image, donor_labels)

				if (
					minority_class_id >= 0
					and minority_donors
					and random.random() < config.copy_paste
				):
					donor_sample = random.choice(minority_donors)
					donor_image, donor_labels = _load_image_and_labels(donor_sample)
					image, labels = _copy_paste_minority(image, labels, donor_image, donor_labels, minority_class_id)

				image = _apply_hsv(image, config.hsv_h, config.hsv_s, config.hsv_v)
				image, labels = _apply_fliplr(image, labels, config.fliplr)
				image, labels = _apply_flipud(image, labels, config.flipud)
				image, labels = _apply_scale(image, labels, config.scale)

				h, w = image.shape[:2]
				labels = _clip_and_filter_boxes(labels, w, h)

				rel_aug_img = sample.rel_image_path.with_name(
					f"{sample.rel_image_path.stem}_aug_{aug_idx}{sample.rel_image_path.suffix}"
				)
				rel_aug_lbl = sample.rel_label_path.with_name(
					f"{sample.rel_label_path.stem}_aug_{aug_idx}.txt"
				)

				out_img_path = out_images_dir / rel_aug_img
				out_lbl_path = out_labels_dir / rel_aug_lbl
				out_img_path.parent.mkdir(parents=True, exist_ok=True)
				out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

				cv2.imwrite(str(out_img_path), image)
				_write_labels(out_lbl_path, labels)
				stats["augmented_images"] += 1

	return stats


def run_preprocess_pipeline(
	raw_root: str | Path = "data/raw",
	resized_root: str | Path = "data/resized",
	augmented_root: str | Path = "data/augmented",
	size: Tuple[int, int] = (640, 640),
	splits: Sequence[str] = ("train", "val", "test"),
	repeats_per_image: int = 1,
	seed: int | None = 42,
) -> Dict[str, Dict[str, int]]:
	"""Convenience wrapper: resize first, then run offline augmentations."""
	resize_stats = resize_dataset(raw_root, resized_root, size=size, splits=splits)
	aug_stats = augment_dataset_offline(
		input_root=resized_root,
		output_root=augmented_root,
		splits=splits,
		config=AugmentConfig(
			mosaic=1.0,
			mixup=0.15,
			copy_paste=0.3,
			hsv_h=0.015,
			hsv_s=0.7,
			hsv_v=0.4,
			flipud=0.0,
			fliplr=0.5,
			scale=0.5,
		),
		repeats_per_image=repeats_per_image,
		copy_originals=True,
		seed=seed,
	)
	return {"resize": resize_stats, "augment": aug_stats}

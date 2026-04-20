from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


def build_model(weights: str = "yolov8s.pt") -> YOLO:
	"""Create a YOLO model initialized from pretrained weights."""
	return YOLO(weights)


def train_yolov8_baseline(
	data_yaml: str | Path = "configs/baseline.yaml",
	weights: str = "yolov8s.pt",
	epochs: int = 30,
	imgsz: int = 640,
	batch: int = 16,
	device: int | str = 0,
	project: str = "runs/baseline",
	name: str = "yolov8s_cadi",
	**extra_train_kwargs: Any,
) -> Dict[str, Any]:
	"""Train YOLOv8s baseline on the configured dataset."""
	data_yaml = Path(data_yaml)
	if not data_yaml.exists():
		raise FileNotFoundError(f"Data config not found: {data_yaml}")

	model = build_model(weights=weights)
	results = model.train(
		data=str(data_yaml),
		epochs=epochs,
		imgsz=imgsz,
		batch=batch,
		device=device,
		project=project,
		name=name,
		**extra_train_kwargs,
	)

	save_dir = getattr(results, "save_dir", None)
	return {
		"weights": weights,
		"data_yaml": str(data_yaml),
		"epochs": epochs,
		"imgsz": imgsz,
		"batch": batch,
		"device": device,
		"project": project,
		"name": name,
		"save_dir": str(save_dir) if save_dir else None,
	}


if __name__ == "__main__":
	summary = train_yolov8_baseline()
	print("Training started with config:")
	print(summary)

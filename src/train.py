from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO


def build_model(weights: str = "yolov8s.pt") -> YOLO:
	"""Create a YOLO model initialized from pretrained weights."""
	return YOLO(weights)


def _to_float_list(values: Any) -> list[float]:
	if values is None:
		return []
	arr = np.asarray(values, dtype=np.float32)
	if arr.size == 0:
		return []
	return arr.reshape(-1).astype(float).tolist()


def export_epoch_metrics(
	run_dir: str | Path,
	out_json_name: str = "epoch_metrics.json",
) -> Dict[str, Any]:
	"""Export YOLO epoch-wise metrics from results.csv into JSON for later analysis."""
	run_dir = Path(run_dir)
	results_csv = run_dir / "results.csv"
	if not results_csv.exists():
		raise FileNotFoundError(f"results.csv not found in run dir: {run_dir}")

	df = pd.read_csv(results_csv)
	columns = [c.strip() for c in df.columns]
	df.columns = columns

	epoch_records = []
	for row in tqdm(df.to_dict(orient="records"), desc="Exporting epoch metrics", unit="epoch"):
		clean_row = {}
		for k, v in row.items():
			if isinstance(v, (np.floating, np.integer)):
				clean_row[k] = float(v)
			else:
				clean_row[k] = v
		epoch_records.append(clean_row)

	out_path = run_dir / out_json_name
	payload = {
		"run_dir": str(run_dir),
		"results_csv": str(results_csv),
		"epochs_logged": len(epoch_records),
		"columns": columns,
		"epochs": epoch_records,
	}
	out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

	return {
		"results_csv": str(results_csv),
		"epoch_metrics_json": str(out_path),
		"epochs_logged": len(epoch_records),
	}


def evaluate_model(
	weights_path: str | Path,
	data_yaml: str | Path,
	split: str = "test",
	save_dir: str | Path | None = None,
	name: str = "eval",
) -> Dict[str, Any]:
	"""Run YOLO evaluation and collect key detection metrics and per-class F1."""
	weights_path = Path(weights_path)
	data_yaml = Path(data_yaml)
	if not weights_path.exists():
		raise FileNotFoundError(f"Weights not found: {weights_path}")
	if not data_yaml.exists():
		raise FileNotFoundError(f"Data config not found: {data_yaml}")

	model = YOLO(str(weights_path))
	val_kwargs: Dict[str, Any] = {
		"data": str(data_yaml),
		"split": split,
		"plots": True,
		"save_json": True,
		"verbose": True,
		"name": name,
	}
	if save_dir is not None:
		val_kwargs["project"] = str(save_dir)

	metrics = model.val(**val_kwargs)
	box = metrics.box

	p_list = _to_float_list(getattr(box, "p", None))
	r_list = _to_float_list(getattr(box, "r", None))
	f1_list = [
		(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
		for p, r in zip(p_list, r_list)
	]

	overall = {
		"map": float(getattr(box, "map", 0.0)),
		"map50": float(getattr(box, "map50", 0.0)),
		"map75": float(getattr(box, "map75", 0.0)),
		"mar": float(np.mean(r_list)) if r_list else 0.0,
		"precision": float(np.mean(p_list)) if p_list else 0.0,
		"recall": float(np.mean(r_list)) if r_list else 0.0,
		"f1": float(np.mean(f1_list)) if f1_list else 0.0,
	}

	# Ultralytics AP is computed across IoU thresholds 0.50:0.95 by default.
	iou_support = {
		"iou_thresholds": [round(x, 2) for x in np.arange(0.5, 1.0, 0.05).tolist()],
		"note": "mAP/mAR are computed across these IoU thresholds.",
	}

	per_class = {
		"precision": p_list,
		"recall": r_list,
		"f1": f1_list,
		"map_per_class": _to_float_list(getattr(box, "maps", None)),
	}

	eval_summary = {
		"weights": str(weights_path),
		"data_yaml": str(data_yaml),
		"split": split,
		"overall": overall,
		"per_class": per_class,
		"iou": iou_support,
	}

	eval_out_path = Path(getattr(metrics, "save_dir", save_dir or "runs")) / "eval_metrics.json"
	eval_out_path.parent.mkdir(parents=True, exist_ok=True)
	eval_out_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

	eval_summary["eval_json"] = str(eval_out_path)
	eval_summary["save_dir"] = str(eval_out_path.parent)
	return eval_summary


def train_yolov8_baseline(
	data_yaml: str | Path = "configs/baseline.yaml",
	weights: str = "yolov8s.pt",
	epochs: int = 30,
	patience: int = 100,
	imgsz: int = 640,
	batch: int = 16,
	device: int | str = 0,
	project: str = "runs/baseline",
	name: str = "yolov8s_cadi",
	save_plots: bool = True,
	export_metrics_json: bool = True,
	**extra_train_kwargs: Any,
) -> Dict[str, Any]:
	"""Train YOLOv8s baseline on the configured dataset and export epoch metrics."""
	data_yaml = Path(data_yaml)
	if not data_yaml.exists():
		raise FileNotFoundError(f"Data config not found: {data_yaml}")

	model = build_model(weights=weights)
	results = model.train(
		data=str(data_yaml),
		epochs=epochs,
		patience=patience,
		imgsz=imgsz,
		batch=batch,
		device=device,
		project=project,
		name=name,
		plots=save_plots,
		verbose=True,
		**extra_train_kwargs,
	)

	save_dir = Path(getattr(results, "save_dir", ""))
	summary: Dict[str, Any] = {
		"weights": weights,
		"data_yaml": str(data_yaml),
		"epochs": epochs,
		"patience": patience,
		"imgsz": imgsz,
		"batch": batch,
		"device": device,
		"project": project,
		"name": name,
		"save_dir": str(save_dir) if str(save_dir) else None,
		"results_csv": str(save_dir / "results.csv") if str(save_dir) else None,
	}

	if export_metrics_json and str(save_dir):
		summary["epoch_metrics"] = export_epoch_metrics(save_dir)

	return summary


def train_and_evaluate_baseline(
	data_yaml: str | Path = "configs/baseline.yaml",
	weights: str = "yolov8s.pt",
	epochs: int = 30,
	patience: int = 100,
	imgsz: int = 640,
	batch: int = 16,
	device: int | str = 0,
	project: str = "runs/baseline",
	name: str = "yolov8s_cadi",
) -> Dict[str, Any]:
	"""Train baseline model then evaluate and save detailed metrics artifacts."""
	train_summary = train_yolov8_baseline(
		data_yaml=data_yaml,
		weights=weights,
		epochs=epochs,
		patience=patience,
		imgsz=imgsz,
		batch=batch,
		device=device,
		project=project,
		name=name,
		save_plots=True,
		export_metrics_json=True,
	)

	run_dir = Path(train_summary["save_dir"])
	best_weights = run_dir / "weights" / "best.pt"
	if not best_weights.exists():
		best_weights = run_dir / "weights" / "last.pt"

	eval_summary = evaluate_model(
		weights_path=best_weights,
		data_yaml=data_yaml,
		split="test",
		save_dir=project,
		name=f"{name}_eval",
	)

	return {
		"train": train_summary,
		"evaluation": eval_summary,
	}


if __name__ == "__main__":
    summary = train_and_evaluate_baseline()
    print("Training and evaluation summary:")
    print(summary)

"""
train_agriyolo.py
==================
AgriYOLO training and evaluation script.

This file is a deliberate structural mirror of src/train.py (the baseline script).
Every public function has an identical signature, identical default hyper-parameters,
identical metric collection, and identical JSON export logic — the ONLY differences
are:

  • build_model() calls build_agriyolo_model() instead of YOLO("yolov8s.pt")
  • project / name defaults reflect the agriyolo run directory

This ensures that all comparisons between baseline and AgriYOLO results are fair:
  – same epochs, batch size, image size, device
  – same augmentation policy  (ultralytics defaults — no custom augmentation)
  – same loss function         (ultralytics default v8 loss)
  – same evaluation split and metric computation
  – same output artefacts      (results.csv, epoch_metrics.json, eval_metrics.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Re-use the helper functions from the baseline script verbatim
from src.train import _to_float_list, export_epoch_metrics, evaluate_model
from src.agriyolo_modules import build_agriyolo_model


# ---------------------------------------------------------------------------
# Model constructor
# ---------------------------------------------------------------------------


def build_model(nc: int = 3) -> Any:
    """Return an AgriYOLO YOLO instance (YOLOv8s + CBAM + BiFPN)."""
    return build_agriyolo_model(nc=nc, verbose=False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_agriyolo(
    data_yaml: str | Path = "configs/baseline.yaml",
    nc: int = 3,
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: int | str = 0,
    project: str = "runs/agriyolo",
    name: str = "agriyolo_cadi",
    save_plots: bool = True,
    export_metrics_json: bool = True,
    **extra_train_kwargs: Any,
) -> Dict[str, Any]:
    """
    Train AgriYOLO on the configured dataset and export epoch-level metrics.

    Parameters are intentionally identical to train_yolov8_baseline() in train.py
    so the two can be called back-to-back in the orchestrator with the same kwargs.

    Augmentation note
    -----------------
    No custom augmentation kwargs are passed.  Ultralytics applies its built-in
    default augmentation policy (identical to what the baseline receives), keeping
    the comparison fair.  Only the model architecture differs.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    model = build_model(nc=nc)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
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
        "model": "AgriYOLO (YOLOv8s + CBAM + BiFPN)",
        "data_yaml": str(data_yaml),
        "epochs": epochs,
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


# ---------------------------------------------------------------------------
# Combined train + evaluate (mirrors train_and_evaluate_baseline exactly)
# ---------------------------------------------------------------------------


def train_and_evaluate_agriyolo(
    data_yaml: str | Path = "configs/baseline.yaml",
    nc: int = 3,
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: int | str = 0,
    project: str = "runs/agriyolo",
    name: str = "agriyolo_cadi",
) -> Dict[str, Any]:
    """
    Train AgriYOLO then evaluate on the test split.

    Return structure is identical to train_and_evaluate_baseline() so the
    orchestrator notebook can use the same print statements for both.

    Returns
    -------
    dict with keys:
        "train"      → training summary (save_dir, results_csv, epoch_metrics)
        "evaluation" → evaluation summary (overall metrics, per-class F1, IoU)
    """
    train_summary = train_agriyolo(
        data_yaml=data_yaml,
        nc=nc,
        epochs=epochs,
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


# ---------------------------------------------------------------------------
# CLI entry point (mirrors train.py __main__ pattern)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = train_and_evaluate_agriyolo()
    print("AgriYOLO training and evaluation summary:")
    print(json.dumps(
        {k: v for k, v in summary["evaluation"]["overall"].items()},
        indent=2,
    ))

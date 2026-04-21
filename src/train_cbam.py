"""
train_cbam.py
=============
CBAM-Only ablation training and evaluation script.

Architectural scope
-------------------
  YOLOv8n backbone  +  CBAM gates after P3 / P4 / P5  +  standard PAN neck (unchanged)

This is a strict ablation of the full AgriYOLO model (train_agriyolo.py).
The BiFPN neck is NOT used here.  The only architectural change from the baseline
is the three C2fWithCBAM wrappers at backbone layers 4, 6, 8.

Fair-comparison guarantees (identical to baseline train.py)
------------------------------------------------------------
  • Same default epochs (30), batch (16), imgsz (640), device (0)
  • No custom augmentation kwargs → ultralytics default policy
  • Same loss function (ultralytics v8 default)
  • export_epoch_metrics / evaluate_model imported directly from src.train
    (same functions, same JSON schema, same metric keys)
  • Same output structure: results.csv, epoch_metrics.json, eval_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# Re-use metric helpers verbatim from baseline — no duplication
from src.train import export_epoch_metrics, evaluate_model
from src.agriyolo_modules import build_cbam_only_model


# ---------------------------------------------------------------------------
# Model constructor
# ---------------------------------------------------------------------------

def build_model(nc: int = 3) -> Any:
    """Return a CBAM-Only YOLO instance (YOLOv8n + CBAM, no BiFPN)."""
    return build_cbam_only_model(nc=nc, verbose=False)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_cbam(
    data_yaml: str | Path = "configs/baseline.yaml",
    nc: int = 3,
    epochs: int = 30,
    patience: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: int | str = 0,
    project: str = "runs/cbam_only",
    name: str = "cbam_cadi",
    save_plots: bool = True,
    export_metrics_json: bool = True,
    **extra_train_kwargs: Any,
) -> Dict[str, Any]:
    """
    Train CBAM-Only model and export epoch-level metrics.

    All parameters mirror train_yolov8_baseline() in src/train.py exactly.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    model = build_model(nc=nc)

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
        "model": "CBAM-Only (YOLOv8n + CBAM, no BiFPN)",
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


# ---------------------------------------------------------------------------
# Combined train + evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate_cbam(
    data_yaml: str | Path = "configs/baseline.yaml",
    nc: int = 3,
    epochs: int = 30,
    patience: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: int | str = 0,
    project: str = "runs/cbam_only",
    name: str = "cbam_cadi",
) -> Dict[str, Any]:
    """
    Train CBAM-Only model then evaluate on test split.

    Return structure is identical to train_and_evaluate_baseline() so the
    orchestrator notebook can use the same print block for all three models.

    Returns
    -------
    dict with keys:
        "train"      → training summary (save_dir, results_csv, epoch_metrics)
        "evaluation" → evaluation summary (overall metrics, per-class F1, IoU)
    """
    train_summary = train_cbam(
        data_yaml=data_yaml,
        nc=nc,
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = train_and_evaluate_cbam()
    print("CBAM-Only training and evaluation summary:")
    print(json.dumps(
        {k: v for k, v in summary["evaluation"]["overall"].items()},
        indent=2,
    ))
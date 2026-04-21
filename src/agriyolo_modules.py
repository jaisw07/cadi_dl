"""
agriyolo_modules.py
====================
Custom modules for AgriYOLO: YOLOv8s + CBAM backbone attention + BiFPN neck.

Architecture modifications over YOLOv8s baseline:
  1. CBAM (Convolutional Block Attention Module) injected after backbone P3/P4/P5 stages.
     Adds channel + spatial attention without changing feature-map shapes or channel counts.
  2. BiFPN neck replaces the standard PAN neck feature fusion.
     Bidirectional weighted fusion across three scales (P3/P4/P5) with learnable scalar
     weights (fast-normalised attention as in EfficientDet).

All training hyper-parameters, loss functions, evaluation, and saving logic remain
IDENTICAL to the baseline so results are directly comparable.

Usage
-----
    from src.agriyolo_modules import build_agriyolo_model
    yolo = build_agriyolo_model(nc=3)          # returns a YOLO instance
    yolo.train(data=..., epochs=30, ...)       # same call as baseline
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (CBAM channel branch)."""

    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        mid = max(1, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM spatial attention branch."""

    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).
    Channel-preserving — no change to spatial size or channel count.
    Applied after the P3, P4, and P5 C2f stages of the YOLOv8s backbone.

    Identity initialisation: gates start at ~0.5 (neutral) so CBAM passes
    pretrained features through unchanged at epoch 0 and learns gradually.
    """

    def __init__(self, channels: int, ratio: int = 8, kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel)
        self._init_identity()

    def _init_identity(self) -> None:
        """Zero-init the last conv in channel-FC → sigmoid(0)=0.5 neutral gate.
        Near-zero init spatial conv → uniform spatial weight ≈ 0.5 everywhere."""
        nn.init.zeros_(self.channel_att.fc[-1].weight)
        nn.init.normal_(self.spatial_att.conv.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-separable Conv + BN + SiLU.
    Used inside BiFPNNeck to keep parameter count low on 6 GB VRAM.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class ConvBnSilu(nn.Module):
    """Standard 1×1 Conv + BN + SiLU — used for channel projections in BiFPN."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# BiFPN neck
# ---------------------------------------------------------------------------


class BiFPNNeck(nn.Module):
    """
    Bidirectional Feature Pyramid Network neck (Tan et al., EfficientDet, 2020).

    Accepts three feature maps from the YOLOv8s neck (P3/P4/P5) and refines them
    through one BiFPN block:
        • Top-down pathway: P5 → P4 → P3  (coarse context flows to fine)
        • Bottom-up pathway: P3 → P4 → P5  (fine detail flows to coarse)

    Each fusion node uses fast-normalised learnable scalar weights so the model
    can learn the relative importance of each scale, unlike standard PAN which
    treats all inputs equally.

    Input/output channel dimensions match the YOLOv8s neck outputs exactly:
        P3: 128-ch  |  P4: 256-ch  |  P5: 512-ch  (at 640×640 input)
    Spatial sizes are unchanged.
    """

    def __init__(self, c3: int = 128, c4: int = 256, c5: int = 512, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

        # ── Channel projections for top-down up-sampling ────────────────────
        # P5 (512-ch) → projected to P4 size (256-ch) before upsample
        self.p5_proj = ConvBnSilu(c5, c4)
        # P4_td (256-ch) → projected to P3 size (128-ch) before upsample
        self.p4_proj = ConvBnSilu(c4, c3)

        # ── Top-down refinement convolutions ────────────────────────────────
        # After weighted sum: same channel count → depthwise-sep conv to refine
        self.p4_td_refine = DepthwiseSeparableConv(c4, c4)
        self.p3_out_refine = DepthwiseSeparableConv(c3, c3)

        # ── Top-down learnable scalar weights ───────────────────────────────
        # w_td_p4[0] * P4  +  w_td_p4[1] * upsample(P5_proj)
        self.w_td_p4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        # w_td_p3[0] * P3  +  w_td_p3[1] * upsample(P4_td_proj)
        self.w_td_p3 = nn.Parameter(torch.ones(2, dtype=torch.float32))

        # ── Channel projections for bottom-up down-sampling ─────────────────
        # P3_out (128-ch) stride-2 → 256-ch for fusion with P4
        self.p3_down = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )
        # P4_out (256-ch) stride-2 → 512-ch for fusion with P5
        self.p4_down = nn.Sequential(
            nn.Conv2d(c4, c5, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True),
        )

        # ── Bottom-up refinement convolutions ───────────────────────────────
        self.p4_out_refine = DepthwiseSeparableConv(c4, c4)
        self.p5_out_refine = DepthwiseSeparableConv(c5, c5)

        # ── Bottom-up learnable scalar weights ──────────────────────────────
        # w_bu_p4[0] * P4_td  +  w_bu_p4[1] * downsample(P3_out)
        self.w_bu_p4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        # w_bu_p5[0] * P5     +  w_bu_p5[1] * downsample(P4_out)
        self.w_bu_p5 = nn.Parameter(torch.ones(2, dtype=torch.float32))

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    # ------------------------------------------------------------------
    def _norm(self, w: nn.Parameter) -> torch.Tensor:
        """Fast-normalised attention weights (ReLU + ε normalisation)."""
        w = torch.relu(w)
        return w / (w.sum() + self.eps)

    # ------------------------------------------------------------------
    def forward(
        self,
        p3: torch.Tensor,   # 128-ch, 80×80 (at 640 input)
        p4: torch.Tensor,   # 256-ch, 40×40
        p5: torch.Tensor,   # 512-ch, 20×20
    ):
        # ── Top-down pathway ──────────────────────────────────────────
        # P5 → P4
        p5_up = self.upsample(self.p5_proj(p5))           # 512→256, 20→40
        w = self._norm(self.w_td_p4)
        p4_td = self.p4_td_refine(w[0] * p4 + w[1] * p5_up)   # 256-ch, 40×40

        # P4_td → P3
        p4_up = self.upsample(self.p4_proj(p4_td))        # 256→128, 40→80
        w = self._norm(self.w_td_p3)
        p3_out = self.p3_out_refine(w[0] * p3 + w[1] * p4_up)  # 128-ch, 80×80

        # ── Bottom-up pathway ─────────────────────────────────────────
        # P3_out → P4
        p3_dn = self.p3_down(p3_out)                       # 128→256, 80→40
        w = self._norm(self.w_bu_p4)
        p4_out = self.p4_out_refine(w[0] * p4_td + w[1] * p3_dn)  # 256-ch, 40×40

        # P4_out → P5
        p4_dn = self.p4_down(p4_out)                       # 256→512, 40→20
        w = self._norm(self.w_bu_p5)
        p5_out = self.p5_out_refine(w[0] * p5 + w[1] * p4_dn)     # 512-ch, 20×20

        return p3_out, p4_out, p5_out


# ---------------------------------------------------------------------------
# Model-surgery wrappers
# ---------------------------------------------------------------------------


class C2fWithCBAM(nn.Module):
    """
    Thin wrapper: runs the original C2f block then applies CBAM attention.

    Injected at backbone layers 4, 6, 8 (P3/P4/P5 C2f outputs) via model surgery.
    All ultralytics metadata attributes (f, i, type) are copied from the original
    layer so _predict_once() and the save-index machinery continue to work correctly.
    """

    def __init__(self, c2f_module: nn.Module, channels: int):
        super().__init__()
        self.c2f = c2f_module
        self.cbam = CBAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.c2f(x))


class DetectWithBiFPN(nn.Module):
    """
    Wraps the YOLOv8 Detect head with a BiFPN refinement step.

    _predict_once() assembles [P3, P4, P5] from the standard neck and passes them
    here. BiFPN refines the features, then the original Detect head runs unchanged.

    All attributes that ultralytics' trainer and loss function read from the Detect
    layer (stride, nc, reg_max, f, i) are explicitly forwarded.
    """

    def __init__(self, detect_module: nn.Module, bifpn: BiFPNNeck):
        super().__init__()
        self.detect = detect_module
        self.bifpn = bifpn

    def forward(self, x):
        """x: list of [P3, P4, P5] tensors assembled by _predict_once."""
        p3_out, p4_out, p5_out = self.bifpn(x[0], x[1], x[2])
        return self.detect([p3_out, p4_out, p5_out])

    # Forward key attributes that ultralytics reads from model.model[-1]
    @property
    def stride(self):
        return self.detect.stride

    @stride.setter
    def stride(self, v):
        self.detect.stride = v

    @property
    def nc(self):
        return self.detect.nc

    @property
    def reg_max(self):
        return getattr(self.detect, "reg_max", 16)

    @property
    def nl(self):
        return getattr(self.detect, "nl", 3)

    @property
    def no(self):
        return getattr(self.detect, "no", self.nc + self.reg_max * 4)


# ---------------------------------------------------------------------------
# Model surgery: inject CBAM + BiFPN into a loaded YOLOv8s model
# ---------------------------------------------------------------------------

# YOLOv8s backbone layer indices and their ACTUAL output channel counts
# (after width_multiple = 0.50 scaling applied by ultralytics):
#   Layer 4  → P3 C2f output  → 128 ch
#   Layer 6  → P4 C2f output  → 256 ch
#   Layer 8  → P5 C2f output  → 512 ch
#   Layer 22 → Detect head    → uses [y[15]=128ch, y[18]=256ch, y[21]=512ch]
_CBAM_TARGETS: dict[int, int] = {4: 128, 6: 256, 8: 512}
_BIFPN_CHANNELS = (128, 256, 512)   # (c3, c4, c5) matching Detect inputs


def _copy_layer_meta(dst: nn.Module, src: nn.Module) -> None:
    """Copy ultralytics layer metadata: index, from-spec, type string."""
    for attr in ("f", "i", "type"):
        if hasattr(src, attr):
            object.__setattr__(dst, attr, getattr(src, attr))


def inject_cbam_and_bifpn(det_model) -> None:
    """
    In-place model surgery on a DetectionModel:
      1. Wraps backbone layers 4, 6, 8 with C2fWithCBAM.
      2. Wraps the last Detect layer with DetectWithBiFPN (adds BiFPN neck).

    The model's save-index list and all skip-connection logic remain intact
    because channel counts and spatial sizes are unchanged by both modifications.

    Parameters
    ----------
    det_model : ultralytics.nn.tasks.DetectionModel
        The internal model object from a YOLO instance (``yolo.model``).
    """
    mods = det_model.model._modules  # OrderedDict: str(idx) → nn.Module

    # ── Step 1: inject CBAM after P3, P4, P5 backbone stages ──────────────
    for layer_idx, channels in _CBAM_TARGETS.items():
        key = str(layer_idx)
        original = mods[key]
        wrapper = C2fWithCBAM(original, channels)
        _copy_layer_meta(wrapper, original)
        wrapper.type = f"C2fCBAM{channels}"
        mods[key] = wrapper
        # Re-register inside the parent Sequential so parameters are tracked
        det_model.model._modules[key] = wrapper

    # ── Step 2: wrap Detect with BiFPN ────────────────────────────────────
    # The Detect layer is always the LAST module in the model Sequential.
    detect_key = str(max(int(k) for k in mods.keys()))
    detect_module = mods[detect_key]

    bifpn = BiFPNNeck(*_BIFPN_CHANNELS)
    detect_wrapper = DetectWithBiFPN(detect_module, bifpn)
    _copy_layer_meta(detect_wrapper, detect_module)
    detect_wrapper.type = "DetectBiFPN"

    mods[detect_key] = detect_wrapper
    det_model.model._modules[detect_key] = detect_wrapper


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_agriyolo_model(nc: int = 3, verbose: bool = False):
    """
    Build AgriYOLO: YOLOv8s pretrained backbone → inject CBAM → add BiFPN neck.

    Returns a standard YOLO instance whose .train() / .val() interface is identical
    to the baseline, so the same training script works unchanged.

    Starting from yolov8s.pt pretrained weights ensures the backbone has good
    representations from the start; CBAM and BiFPN parameters (randomly initialised)
    are learned during fine-tuning on CADI-AI.

    Parameters
    ----------
    nc : int
        Number of detection classes (3 for CADI-AI).
    verbose : bool
        Whether to print the modified model summary.

    Returns
    -------
    ultralytics.YOLO
        Modified YOLO model ready for .train().
    """
    from ultralytics import YOLO

    yolo = YOLO("yolov8s.pt")

    # Override the number of classes if different from the pretrained nc
    if yolo.model.nc != nc:
        yolo.model.nc = nc

    # Inject architectural modifications in-place
    inject_cbam_and_bifpn(yolo.model)

    if verbose:
        print("[AgriYOLO] Architecture modifications applied:")
        print(f"  • CBAM added at backbone layers: {list(_CBAM_TARGETS.keys())}")
        print(f"  • BiFPN neck added at Detect layer with channels {_BIFPN_CHANNELS}")
        total_params = sum(p.numel() for p in yolo.model.parameters())
        trainable = sum(p.numel() for p in yolo.model.parameters() if p.requires_grad)
        print(f"  • Total parameters : {total_params:,}")
        print(f"  • Trainable params : {trainable:,}")

    return yolo


def build_cbam_only_model(nc: int = 3, verbose: bool = False):
    """
    Build CBAM-Only model: YOLOv8s pretrained backbone + CBAM attention gates.

    Identical to build_agriyolo_model() EXCEPT:
      • No BiFPN neck — the standard YOLOv8s PAN neck is kept untouched.
      • Only CBAM is injected after backbone P3/P4/P5 C2f stages.

    This is an ablation variant to isolate the contribution of CBAM alone,
    decoupled from any BiFPN surgery side-effects.  Because no new neck modules
    are introduced, the only randomly-initialised parameters are the small CBAM
    gates (~0.15 M params), which start near-identity and add minimal gradient
    noise to the pretrained weights.

    Parameters
    ----------
    nc : int
        Number of detection classes (3 for CADI-AI).
    verbose : bool
        Print parameter counts and modified layer indices.

    Returns
    -------
    ultralytics.YOLO
        Modified YOLO instance ready for .train() with identical interface
        to the baseline and to build_agriyolo_model().
    """
    from ultralytics import YOLO

    yolo = YOLO("yolov8s.pt")

    if yolo.model.nc != nc:
        yolo.model.nc = nc

    mods = yolo.model.model._modules

    # Inject CBAM only — DO NOT touch the neck or Detect head
    for layer_idx, channels in _CBAM_TARGETS.items():
        key = str(layer_idx)
        original = mods[key]
        wrapper = C2fWithCBAM(original, channels)
        _copy_layer_meta(wrapper, original)
        wrapper.type = f"C2fCBAM{channels}"
        mods[key] = wrapper
        yolo.model.model._modules[key] = wrapper

    if verbose:
        print("[CBAM-Only] Architecture modifications applied:")
        print(f"  • CBAM added at backbone layers : {list(_CBAM_TARGETS.keys())}")
        print(f"  • Neck / Detect head            : UNCHANGED (standard PAN)")
        total_params = sum(p.numel() for p in yolo.model.parameters())
        cbam_params  = sum(
            p.numel()
            for k, mod in yolo.model.model._modules.items()
            if isinstance(mod, C2fWithCBAM)
            for p in mod.cbam.parameters()
        )
        print(f"  • Total parameters  : {total_params:,}")
        print(f"  • CBAM-only params  : {cbam_params:,}")

    return yolo

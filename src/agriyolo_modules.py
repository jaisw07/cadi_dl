from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CBAM (UPGRADED - RESIDUAL + STRONGER)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
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

        # 🔥 less conservative init (IMPORTANT)
        nn.init.normal_(self.fc[-1].weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """🔥 Residual CBAM (key upgrade)"""

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        attn = self.ca(x) * self.sa(x)
        return x + x * attn   # 🔥 RESIDUAL (critical)


# ---------------------------------------------------------------------------
# CONVS
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class ConvBnSilu(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# BiFPN (UPGRADED: DOUBLE STACK + CBAM OUTPUT)
# ---------------------------------------------------------------------------

class BiFPNBlock(nn.Module):
    def __init__(self, c3=128, c4=256, c5=512, eps=1e-4):
        super().__init__()
        self.eps = eps

        self.p5_proj = ConvBnSilu(c5, c4)
        self.p4_proj = ConvBnSilu(c4, c3)

        self.p4_td_refine = DepthwiseSeparableConv(c4, c4)
        self.p3_out_refine = DepthwiseSeparableConv(c3, c3)

        self.w_td_p4 = nn.Parameter(torch.ones(2))
        self.w_td_p3 = nn.Parameter(torch.ones(2))

        self.p3_down = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )

        self.p4_down = nn.Sequential(
            nn.Conv2d(c4, c5, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True),
        )

        self.p4_out_refine = DepthwiseSeparableConv(c4, c4)
        self.p5_out_refine = DepthwiseSeparableConv(c5, c5)

        self.w_bu_p4 = nn.Parameter(torch.ones(2))
        self.w_bu_p5 = nn.Parameter(torch.ones(2))

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _norm(self, w):
        w = torch.relu(w)
        return w / (w.sum() + self.eps)

    def forward(self, p3, p4, p5):
        # top-down
        p5_up = self.upsample(self.p5_proj(p5))
        w = self._norm(self.w_td_p4)
        p4_td = self.p4_td_refine(w[0]*p4 + w[1]*p5_up)

        p4_up = self.upsample(self.p4_proj(p4_td))
        w = self._norm(self.w_td_p3)
        p3_out = self.p3_out_refine(w[0]*p3 + w[1]*p4_up)

        # bottom-up
        p3_dn = self.p3_down(p3_out)
        w = self._norm(self.w_bu_p4)
        p4_out = self.p4_out_refine(w[0]*p4_td + w[1]*p3_dn)

        p4_dn = self.p4_down(p4_out)
        w = self._norm(self.w_bu_p5)
        p5_out = self.p5_out_refine(w[0]*p5 + w[1]*p4_dn)

        return p3_out, p4_out, p5_out


class BiFPNNeck(nn.Module):
    """🔥 Double BiFPN + CBAM refinement"""

    def __init__(self):
        super().__init__()
        self.b1 = BiFPNBlock()
        self.b2 = BiFPNBlock()

        # 🔥 NEW: CBAM on outputs
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(256)
        self.cbam5 = CBAM(512)

    def forward(self, p3, p4, p5):
        p3, p4, p5 = self.b1(p3, p4, p5)
        p3, p4, p5 = self.b2(p3, p4, p5)

        # 🔥 critical improvement
        p3 = self.cbam3(p3)
        p4 = self.cbam4(p4)
        p5 = self.cbam5(p5)

        return p3, p4, p5


# ---------------------------------------------------------------------------
# WRAPPERS
# ---------------------------------------------------------------------------

class C2fWithCBAM(nn.Module):
    """🔥 Residual integration instead of plain overwrite"""

    def __init__(self, c2f_module: nn.Module, channels: int):
        super().__init__()
        self.c2f = c2f_module
        self.cbam = CBAM(channels)

    def forward(self, x):
        y = self.c2f(x)
        return y + self.cbam(y)   # 🔥 stronger than before


class DetectWithBiFPN(nn.Module):
    def __init__(self, detect_module, bifpn):
        super().__init__()
        self.detect = detect_module
        self.bifpn = bifpn

    def forward(self, x):
        p3, p4, p5 = self.bifpn(x[0], x[1], x[2])
        return self.detect([p3, p4, p5])

    @property
    def stride(self):
        return self.detect.stride

    @stride.setter
    def stride(self, v):
        self.detect.stride = v

    @property
    def nc(self):
        return self.detect.nc


# ---------------------------------------------------------------------------
# INJECTION
# ---------------------------------------------------------------------------

_CBAM_TARGETS = {4: 128, 6: 256, 8: 512}


def inject_cbam_and_bifpn(det_model):
    mods = det_model.model._modules

    for idx, ch in _CBAM_TARGETS.items():
        key = str(idx)
        original = mods[key]
        mods[key] = C2fWithCBAM(original, ch)
        det_model.model._modules[key] = mods[key]

    detect_key = str(max(int(k) for k in mods.keys()))
    detect_module = mods[detect_key]

    wrapper = DetectWithBiFPN(detect_module, BiFPNNeck())
    mods[detect_key] = wrapper
    det_model.model._modules[detect_key] = wrapper


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------

def build_agriyolo_model(nc=3, verbose=False):
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")
    if yolo.model.nc != nc:
        yolo.model.nc = nc

    inject_cbam_and_bifpn(yolo.model)
    return yolo


def build_cbam_only_model(nc=3):
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")
    if yolo.model.nc != nc:
        yolo.model.nc = nc

    mods = yolo.model.model._modules

    for idx, ch in _CBAM_TARGETS.items():
        key = str(idx)
        original = mods[key]
        mods[key] = C2fWithCBAM(original, ch)
        yolo.model.model._modules[key] = mods[key]

    return yolo
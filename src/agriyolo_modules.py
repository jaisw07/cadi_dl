from __future__ import annotations
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f


# ---------------------------------------------------------------------------
# CBAM (MATCH ARCH 2 EXACTLY)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(
            self.mlp(self.avg_pool(x)) +
            self.mlp(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.act(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """ARCH 2 EXACT VERSION"""

    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        attn = self.channel(x) * self.spatial(x)
        return x + x * attn   # residual


# ---------------------------------------------------------------------------
# 🔥 TRUE C2f + CBAM (CRITICAL FIX)
# ---------------------------------------------------------------------------

class C2fCBAM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.cbam = CBAM(c2)

    def forward(self, x):
        y = super().forward(x)
        return self.cbam(y)


# ---------------------------------------------------------------------------
# BiFPN (LIGHT — MATCH ARCH 2 STYLE)
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BiFPN(nn.Module):
    def __init__(self, c3=128, c4=256, c5=512):
        super().__init__()

        self.p5_up = ConvBNAct(c5, c4)
        self.p4_up = ConvBNAct(c4, c3)

        self.p3_down = nn.Conv2d(c3, c4, 3, stride=2, padding=1)
        self.p4_down = nn.Conv2d(c4, c5, 3, stride=2, padding=1)

        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, p3, p4, p5):
        # top-down
        p4_td = self.p4_up(p4) + self.upsample(self.p5_up(p5))
        p3_out = p3 + self.upsample(p4_td)

        # bottom-up
        p4_out = p4_td + self.p3_down(p3_out)
        p5_out = p5 + self.p4_down(p4_out)

        return p3_out, p4_out, p5_out


# ---------------------------------------------------------------------------
# DETECT WRAPPER (UNCHANGED LOGIC)
# ---------------------------------------------------------------------------

class DetectWithBiFPN(nn.Module):
    def __init__(self, detect_module):
        super().__init__()
        self.detect = detect_module
        self.bifpn = BiFPN()

    def forward(self, x):
        p3, p4, p5 = self.bifpn(x[0], x[1], x[2])
        return self.detect([p3, p4, p5])

    @property
    def stride(self):
        return self.detect.stride

    @property
    def nc(self):
        return self.detect.nc


# ---------------------------------------------------------------------------
# 🔥 TRUE REPLACEMENT (NOT WRAPPING)
# ---------------------------------------------------------------------------

def inject_cbam_and_bifpn(model):
    mods = model.model._modules

    targets = {4: 128, 6: 256, 8: 512}

    # 🔥 Instead of rebuilding, wrap safely
    for idx, ch in targets.items():
        key = str(idx)
        old = mods[key]

        # attach CBAM dynamically
        old.cbam = CBAM(ch)

        # override forward
        def new_forward(self, x):
            y = super(type(self), self).forward(x)
            return self.cbam(y)

        old.forward = new_forward.__get__(old, type(old))

    # Detect + BiFPN (same as before)
    detect_key = str(max(int(k) for k in mods.keys()))
    detect_module = mods[detect_key]

    mods[detect_key] = DetectWithBiFPN(detect_module)
    model.model._modules[detect_key] = mods[detect_key]


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


def build_cbam_only_model(nc=3, verbose=False):
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")

    mods = yolo.model.model._modules
    targets = {4: 128, 6: 256, 8: 512}

    for idx, ch in targets.items():
        key = str(idx)
        old = mods[key]

        new = C2fCBAM(
            old.c1,
            old.c2,
            n=old.n,
            shortcut=old.shortcut,
            g=old.g,
            e=old.e
        )

        new.load_state_dict(old.state_dict(), strict=False)

        mods[key] = new
        yolo.model.model._modules[key] = new

    return yolo
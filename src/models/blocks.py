import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.drop(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat


def _align(x, ref):
    """Bilinear-resize x to match ref's spatial size (guards against odd-size
    rounding mismatches between encoder stages and decoder upsampling)."""
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class UpBlock(nn.Module):
    """Upsample, concatenate with the matching encoder skip connection, fuse
    with a DoubleConv. Carrying the encoder's high-resolution features
    straight into the decoder at every stage is what keeps the predicted
    heatmap sharp -- without it, a decoder upsampling purely from a deep,
    heavily-downsampled bottleneck has no fine-grained signal left to recover
    and tends to degrade into flat, near-uniform output."""

    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = _align(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018 - Attention U-Net).
    Uses the coarser decoder signal to suppress irrelevant background in the
    finer encoder skip connection before fusing them -- this is what lets
    Attention U-Net stay sharp on cluttered satellite backgrounds where plain
    U-Net skip connections leak background noise through."""

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=True), nn.BatchNorm2d(inter_ch)
        )
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=True), nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        gate_r = _align(self.w_gate(gate), skip)
        attn = self.relu(gate_r + self.w_skip(skip))
        attn = self.psi(attn)
        return skip * attn


class AttnUpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(gate_ch=out_ch, skip_ch=skip_ch, inter_ch=max(out_ch // 2, 8))
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = _align(x, skip)
        gated_skip = self.attn(x, skip)
        x = torch.cat([x, gated_skip], dim=1)
        return self.conv(x)

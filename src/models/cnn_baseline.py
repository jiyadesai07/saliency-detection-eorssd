import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNReLU


class CNNBaseline(nn.Module):
    """The simplest model in the comparison: a plain convolutional
    encoder-decoder with NO skip connections. It serves as the baseline
    against which the benefit of skip connections (U-Net) and pretrained
    encoders + attention (below) can be measured directly in the results
    table, rather than just asserted."""

    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc = nn.Sequential(
            ConvBNReLU(in_ch, c), ConvBNReLU(c, c), nn.MaxPool2d(2),
            ConvBNReLU(c, c * 2), ConvBNReLU(c * 2, c * 2), nn.MaxPool2d(2),
            ConvBNReLU(c * 2, c * 4), ConvBNReLU(c * 4, c * 4), nn.MaxPool2d(2),
            ConvBNReLU(c * 4, c * 8), ConvBNReLU(c * 8, c * 8),
        )
        self.dec = nn.Sequential(
            ConvBNReLU(c * 8, c * 4),
            ConvBNReLU(c * 4, c * 2),
            ConvBNReLU(c * 2, c),
        )
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        feat = self.enc(x)
        feat = self.dec(feat)
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(feat)

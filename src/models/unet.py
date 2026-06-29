import torch.nn as nn

from .blocks import DownBlock, DoubleConv, UpBlock


class UNet(nn.Module):
    """Plain U-Net trained from scratch: a 5-level encoder/decoder with
    BatchNorm + Dropout and skip connections at every stage, serving as the
    from-scratch counterpart to the pretrained-encoder models below."""

    def __init__(self, in_ch=3, base_ch=64, dropout=0.1):
        super().__init__()
        c = base_ch
        self.down1 = DownBlock(in_ch, c, dropout)
        self.down2 = DownBlock(c, c * 2, dropout)
        self.down3 = DownBlock(c * 2, c * 4, dropout)
        self.down4 = DownBlock(c * 4, c * 8, dropout)
        self.bottleneck = DoubleConv(c * 8, c * 16, dropout)

        self.up1 = UpBlock(c * 16, c * 8, c * 8, dropout)
        self.up2 = UpBlock(c * 8, c * 4, c * 4, dropout)
        self.up3 = UpBlock(c * 4, c * 2, c * 2, dropout)
        self.up4 = UpBlock(c * 2, c, c, dropout)

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x, s4 = self.down4(x)
        x = self.bottleneck(x)

        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.head(x)  # logits, shape (B, 1, H, W)

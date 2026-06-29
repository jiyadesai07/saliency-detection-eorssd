import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from .blocks import UpBlock


class VGGUNet(nn.Module):
    """VGG16-BN encoder (ImageNet-pretrained) with a proper U-Net decoder that
    concatenates skip connections from every encoder stage, so the decoder
    has a path back to high-resolution detail at every level of upsampling
    rather than just the 1/32-resolution bottleneck."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16_bn(weights=weights).features

        self.stage1 = features[0:6]    # -> 64ch,  full res
        self.pool1 = features[6:7]
        self.stage2 = features[7:13]   # -> 128ch, 1/2
        self.pool2 = features[13:14]
        self.stage3 = features[14:23]  # -> 256ch, 1/4
        self.pool3 = features[23:24]
        self.stage4 = features[24:33]  # -> 512ch, 1/8
        self.pool4 = features[33:34]
        self.stage5 = features[34:43]  # -> 512ch, 1/16
        self.pool5 = features[43:44]   # -> 512ch, 1/32 (bottleneck)

        self.up1 = UpBlock(512, 512, 512, dropout)
        self.up2 = UpBlock(512, 512, 256, dropout)
        self.up3 = UpBlock(256, 256, 128, dropout)
        self.up4 = UpBlock(128, 128, 64, dropout)
        self.up5 = UpBlock(64, 64, 64, dropout)

        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.stage1(x)
        x = self.pool1(s1)
        s2 = self.stage2(x)
        x = self.pool2(s2)
        s3 = self.stage3(x)
        x = self.pool3(s3)
        s4 = self.stage4(x)
        x = self.pool4(s4)
        s5 = self.stage5(x)
        x = self.pool5(s5)

        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        return self.head(x)

import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from .blocks import DoubleConv, UpBlock


class ResNetUNet(nn.Module):
    """ResNet34 encoder (ImageNet-pretrained) with a proper U-Net decoder,
    wiring up skip connections from every encoder stage so the decoder has a
    path back to high-resolution spatial detail rather than upsampling purely
    from the 1/32-resolution bottleneck."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64ch, 1/2
        self.maxpool = backbone.maxpool                                         # -> 1/4
        self.layer1 = backbone.layer1  # 64ch,  1/4
        self.layer2 = backbone.layer2  # 128ch, 1/8
        self.layer3 = backbone.layer3  # 256ch, 1/16
        self.layer4 = backbone.layer4  # 512ch, 1/32 (bottleneck)

        self.up1 = UpBlock(512, 256, 256, dropout)
        self.up2 = UpBlock(256, 128, 128, dropout)
        self.up3 = UpBlock(128, 64, 64, dropout)
        self.up4 = UpBlock(64, 64, 64, dropout)  # skip = stem output

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = DoubleConv(32, 32, dropout)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        stem = self.stem(x)        # 1/2
        x = self.maxpool(stem)     # 1/4
        s1 = self.layer1(x)        # 1/4
        s2 = self.layer2(s1)       # 1/8
        s3 = self.layer3(s2)       # 1/16
        x = self.layer4(s3)        # 1/32 bottleneck

        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, stem)      # back to 1/2 res

        x = self.final_up(x)       # -> full res
        x = self.final_conv(x)
        return self.head(x)

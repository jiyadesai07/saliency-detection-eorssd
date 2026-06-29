import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from .blocks import AttnUpBlock, DoubleConv

"""Attention U-Net (Oktay et al., 2018) targets the main failure mode this
dataset has: cluttered satellite backgrounds (roads, fields, shadows) leaking
into the skip connections and diluting the salient region. Each attention
gate learns to weight the encoder's skip features by relevance to the
decoder's current focus before they're concatenated in -- a natural
extension of the plain U-Net/ResNet-UNet/VGG-UNet models above."""


class AttentionResNetUNet(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64ch, 1/2
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64ch,  1/4
        self.layer2 = backbone.layer2  # 128ch, 1/8
        self.layer3 = backbone.layer3  # 256ch, 1/16
        self.layer4 = backbone.layer4  # 512ch, 1/32 (bottleneck)

        self.up1 = AttnUpBlock(512, 256, 256, dropout)
        self.up2 = AttnUpBlock(256, 128, 128, dropout)
        self.up3 = AttnUpBlock(128, 64, 64, dropout)
        self.up4 = AttnUpBlock(64, 64, 64, dropout)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = DoubleConv(32, 32, dropout)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        stem = self.stem(x)
        x = self.maxpool(stem)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        x = self.layer4(s3)

        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, stem)

        x = self.final_up(x)
        x = self.final_conv(x)
        return self.head(x)

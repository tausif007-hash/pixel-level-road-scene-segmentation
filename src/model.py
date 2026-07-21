"""
model.py
Model architecture definitions: U-Net building blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Building blocks
class DoubleConv(nn.Module):
    """ Conv → BN → ReLU → Conv → BN → ReLU """

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """MaxPool ÷2 → DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    """Bilinear ×2 → concat skip → DoubleConv"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([skip, x], dim=1))

# Models
class UNet(nn.Module):
    """
    Vanilla U-Net.
    Input  : [B, 3, H, W]
    Output : [B, num_classes, H, W]  (raw logits)
    """

    def __init__(self, in_channels=3, num_classes=11, base_filters=64):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = EncoderBlock(f, f * 2)
        self.enc3 = EncoderBlock(f * 2, f * 4)
        self.enc4 = EncoderBlock(f * 4, f * 8)
        self.bottleneck = EncoderBlock(f * 8, f * 16)
        self.dec4 = DecoderBlock(f * 16, f * 8, f * 8)
        self.dec3 = DecoderBlock(f * 8, f * 4, f * 4)
        self.dec2 = DecoderBlock(f * 4, f * 2, f * 2)
        self.dec1 = DecoderBlock(f * 2, f, f)
        self.head = nn.Conv2d(f, num_classes, kernel_size=1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)


class UNetSmall(UNet):
    """Half-width U-Net — for fast CPU/low-VRAM testing."""

    def __init__(self, in_channels=3, num_classes=11):
        super().__init__(in_channels, num_classes, base_filters=32)


class UNetResNet34(nn.Module):

    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights
        bb = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        print(f'  ResNet-34 backbone loaded (pretrained={pretrained})')

        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)  # /2
        self.pool = bb.maxpool                                 # /4
        self.enc1 = bb.layer1   # /4
        self.enc2 = bb.layer2   # /8
        self.enc3 = bb.layer3   # /16
        self.enc4 = bb.layer4   # /32

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)
        p = self.pool(e0)
        e1 = self.enc1(p)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        return self.head(self.up(d1))


def build_model(arch='unet_resnet34', num_classes=11, pretrained=True):
    """Factory function — builds a model from an architecture name string."""
    arch = arch.lower()
    if arch == 'unet':
        return UNet(num_classes=num_classes)
    elif arch == 'unet_small':
        return UNetSmall(num_classes=num_classes)
    elif arch == 'unet_resnet34':
        return UNetResNet34(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f'Unknown arch: {arch}')

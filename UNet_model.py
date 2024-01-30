import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self: 'DoubleConv', in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self: 'DoubleConv', x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self: 'Down', in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self: 'Down', x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self: 'Up', in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self: 'Up', x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1    = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1    = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x     = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self: 'OutConv', in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self: 'OutConv', x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channel: int, dim_feature: int, out_channel: int, bilinear: bool = False, out_activation: str = 'tanh') -> None:
        super(UNet, self).__init__()
        
        self.in_channel     = in_channel
        self.dim_feature    = dim_feature
        self.out_channel    = out_channel
        self.bilinear       = bilinear
        self.out_activation = out_activation
        self.criterion      = nn.MSELoss(reduction='sum')

        self.inc   = DoubleConv(in_channel, dim_feature * 2)
        self.down1 = Down(dim_feature * 2, dim_feature * 4)
        self.down2 = Down(dim_feature * 4, dim_feature * 8)
        self.down3 = Down(dim_feature * 8, dim_feature * 16)
        
        factor     = 2 if bilinear else 1
        # self.down4 = Down(dim_feature * 8, dim_feature * 16)
        self.down4 = Down(dim_feature * 16, dim_feature * 32 // factor)
        self.up1   = Up(dim_feature * 32, dim_feature * 16 // factor, bilinear)
        self.up2   = Up(dim_feature * 16, dim_feature * 8 // factor, bilinear)
        self.up3   = Up(dim_feature * 8, dim_feature * 4 // factor, bilinear)
        self.up4   = Up(dim_feature * 4, dim_feature * 2, bilinear)
        self.outc  = OutConv(dim_feature * 2, out_channel)

        if self.out_activation == 'tanh':
            self.output = nn.Tanh()

        elif self.out_activation == 'sigmoid':
            self.output = nn.Sigmoid()
        
    def forward(self: 'UNet', x: torch.Tensor) -> torch.Tensor:
        x1     = self.inc(x)
        x2     = self.down1(x1)
        x3     = self.down2(x2)
        x4     = self.down3(x3)
        x5     = self.down4(x4)
        x      = self.up1(x5, x4)
        x      = self.up2(x, x3)
        x      = self.up3(x, x2)
        x      = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.output(logits)

        return logits

    def compute_loss(self: 'UNet', prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_recon = self.criterion(prediction, target)

        return loss_recon
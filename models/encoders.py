import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import SoftPositionEmbed
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)            
        )

    def forward(self, x):
        return self.downsample(x)
    
class CNNEncoder(nn.Module):
    def __init__(self, slot_dim, num_channels=1):
        super(CNNEncoder, self).__init__()
        self.slot_dim = slot_dim
        self.num_channels = num_channels

        self.conv1 = ConvBlock(num_channels, 16) # shape (batch_size, 16, 224, 224)
        self.conv2 = Downsample(16, 32)     # shape (batch_size, 32, 112, 112)
        self.conv3 = Downsample(32, 64)     # shape (batch_size, 64, 56, 56)
        self.conv4 = Downsample(64, 128)    # shape (batch_size, 128, 28, 28)
        self.conv5 = Downsample(128, self.slot_dim)   # shape (batch_size, 256, 14, 14)

    def forward(self, x):
        in_res = x.shape[-1]
        end_res = in_res // 2**4
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # flatten image
        x = x.view(-1, end_res**2, self.slot_dim)

        return x
    

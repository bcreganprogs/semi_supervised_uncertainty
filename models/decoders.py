import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_local.utils import SoftPositionEmbed
import math
from typing import Callable, Dict, Optional, Tuple, Union
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, end_relu=True):
        super(ConvBlock, self).__init__()

        if end_relu:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels, end_relu=False),
        )

        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        self.output = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):

        skip = x

        x = self.conv(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        return self.output(x + skip)
        # return x + skip

class ECA(nn.Module):
    """Version from from https://wandb.ai/diganta/ECANet-sweep/reports/Efficient-Channel-Attention--VmlldzozNzgwOTE
    Constructs a ECA module.


    Args:
        channels: Number of channels in the input tensor
        b: Hyper-parameter for adaptive kernel size formulation. Default: 1
        gamma: Hyper-parameter for adaptive kernel size formulation. Default: 2 
    """
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out

    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    """From https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py"""
    def __init__(self, in_chans, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_chans, in_chans // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_chans // 16, in_chans, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """From https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, size=None):
        super(Upsample, self).__init__()

        # if size is not None:
        #     self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)
        # else:
        #     self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
        )

        self.conv = ConvBlock(in_channels, out_channels, end_relu=True)      #ResBlock(in_channels, out_channels)


    def forward(self, x):
        
        x = self.upsample(x)
        x = self.conv(x)

        return x  
    
class CNNDecoder(nn.Module):
    """Additive CNN Decoder."""
    def __init__(self, dim, slot_dim, num_slots, num_classes, resolution=224,
                 image_chans=1, decoder_type='slot_specific'):
        super().__init__()
        self.dim = dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.resolution = resolution
        self.image_chans = image_chans
        self.decoder_type = decoder_type
        self.decoder_initial_size = (8, 8)
        self.decoder = self.make_decoder()
        self.training = True

    def make_decoder(self):
        layers = [
            ConvBlock(self.slot_dim, self.dim, end_relu=True), # 8, 8
            Upsample(self.dim, self.dim, size=(16, 16)), # 16, 16
            Upsample(self.dim, self.dim, size=(32, 32)), # 32, 32
            Upsample(self.dim, self.dim, size=(64, 64)), # 64, 64
            Upsample(self.dim, 64, size=(128, 128)), # 128, 128
        ]
        if self.resolution > 128 and self.resolution <= 256:
            layers.append(Upsample(64, 64, size=(self.resolution, self.resolution))) # 256, 256
        elif self.resolution > 128:
            layers.append(Upsample(64, 64, size=(256, 256)))
        if self.resolution > 256:
            layers.append(Upsample(64, 64, size=(self.resolution, self.resolution))) # 512, 512
       
        layers.append(nn.Conv2d(64, 64, kernel_size=5, padding=2)) # no batch norm
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, self.image_chans + 1, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, num_slots, slot_dim, init_height, init_width = x.size()
    
        x = x.view(-1, slot_dim, init_height, init_width)
        
        x = self.decoder(x)

        # crop to match resolution
        x = x[:, :, :self.resolution, :self.resolution]

        x = x.view(batch_size, num_slots, self.image_chans + 1, self.resolution, self.resolution)

        return x
    
# Code taken from spot https://github.com/gkakogeorgiou/spot/blob/master/mlp.py and is based on 
# https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl
class MlpDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        hidden_features: Dimension of hidden layers.
    """

    def __init__(self, object_dim, output_dim, num_patches, hidden_features = 2048):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, object_dim) * 0.02)
        self.decoder = build_mlp(object_dim, output_dim + 1, hidden_features)

    def forward(self, encoder_output):

        initial_shape = encoder_output.shape[:-1]
        encoder_output = encoder_output.flatten(0, -2)

        encoder_output = encoder_output.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT
        object_features = encoder_output + self.pos_embed

        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)
        
        return reconstruction, masks
    
    
def build_mlp(input_dim = int, output_dim = int, hidden_features = 2048, n_hidden_layers = 3):
    
    layers = []
    current_dim = input_dim
    features = [hidden_features]*n_hidden_layers

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.ReLU(inplace=True))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)

    return nn.Sequential(*layers)

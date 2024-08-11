import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import SoftPositionEmbed

from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
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
        self.conv4 = ConvBlock(64, 128)    # shape (batch_size, 128, 28, 28)
        self.conv5 = Downsample(128, 256)   # shape (batch_size, 256, 14, 14)
        self.conv6 = ConvBlock(256, self.slot_dim)

    def forward(self, x, mask_ratio=0.0):
        # if x is not a tensor, make random one
        if not isinstance(x, torch.Tensor):
            return x
        in_res = x.shape[-1]
        end_res = in_res // 2**3
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # flatten image
        x = x.view(-1, end_res**2, self.slot_dim)

        return x
    
def get_resnet34_encoder():
    model = resnet34(weights=None)#ResNet34_Weights.DEFAULT)

    # Save the original weights and bias
    original_conv = model.conv1
    original_weight = original_conv.weight.clone()
    original_bias = original_conv.bias.clone() if original_conv.bias is not None else None

    # Create a new convolutional layer with 1 input channel
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Initialize the new layer with the average of the original weights across the channel dimension
    new_conv.weight.data = original_weight.sum(dim=1, keepdim=True)

    # If there was a bias, keep it
    if original_bias is not None:
        new_conv.bias = nn.Parameter(original_bias)

    # Replace the first layer
    model.conv1 = new_conv

    return model

class ResNet34_8x8(nn.Module):
    def __init__(self, base_model):
        super(ResNet34_8x8, self).__init__()
        self.features = nn.Sequential(
            *list(base_model.children())[:7]
        )

        self.features_1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,

        )
        self.features_2 = nn.Sequential(
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
    
    def forward(self, x):
        x1 = self.features_1(x) # shape (batch_size, 64, 32, 32)
        x2 = self.features_2(x1) # shape (batch_size, 256, 8, 8)
      
        return x2, x1

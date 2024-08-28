import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_local.utils import SoftPositionEmbed

from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet18, ResNet18_Weights

    

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
    
def get_resnet34_encoder(pretrained=True):
    if pretrained:
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
    else:
        model = resnet34(weights=None)

    return model

def get_resnet18_encoder(pretrained=True):
    if not pretrained:
        model = resnet18(weights=None)
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

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
            base_model.layer2,
            

        )
        self.features_2 = nn.Sequential(
            base_model.layer3,
            base_model.layer4,
        )
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x1 = self.features_1(x) # shape (batch_size, 64, 32, 32)
        x2 = self.features_2(x1) # shape (batch_size, 256, 8, 8)
      
        return x1

class ResNet18(nn.Module):
    def __init__(self, base_model):
        super(ResNet18, self).__init__()
        self.features = nn.Sequential(
            *list(base_model.children())[:7]
        )

        self.features_1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,

        )
        self.features_2 = nn.Sequential(
            
            base_model.layer4,
        )
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x1 = self.features_1(x) # shape (batch_size, 64, 32, 32)
        # x2 = self.features_2(x1) # shape (batch_size, 256, 8, 8)
      
        return x1
    
class DinoViT_16(nn.Module):
    def __init__(self, num_channels=1):
        super(DinoViT_16, self).__init__()
        
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')


    def forward(self, x):
        
        # if x only has one channel, expand it to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.model.prepare_tokens(x)

        for blk in self.model.blocks:
            x = blk(x)

        x = x[:, 1:] # remove the cls token

        return x
    
class DinoViTB_16(nn.Module):
    def __init__(self, num_channels=1):
        super(DinoViTB_16, self).__init__()
        
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    def forward(self, x):
        
        # if x only has one channel, expand it to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.model.prepare_tokens(x)

        for blk in self.model.blocks:
            x = blk(x)

        x = x[:, 1:] # remove the cls token

        return x

class Dinov2ViT_14(nn.Module):
    def __init__(self, num_channels=1):
        super(Dinov2ViT_14, self).__init__()
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    def forward(self, x):
        
        # if x only has one channel, expand it to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)      
        x = self.get_patch_embeddings(x)[0]
        x = x[:, 1:, :] # remove the cls token

        return x
    
    def get_patch_embeddings(self, x):
        intermediate_outputs = []
        hooks = []
    
        def hook(module, input, output):
            intermediate_outputs.append(output)
        
        for block in self.model.blocks[-1:]:
            handle = block.register_forward_hook(hook)
            hooks.append(handle)
        
        _ = self.model(x)

        # remove the hooks
        for h in hooks:
            h.remove()
        
        return intermediate_outputs
    
class DinoViT_8(nn.Module):
    def __init__(self, num_channels=1):
        super(DinoViT_8, self).__init__()
        
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')


    def forward(self, x):
        
        # if x only has one channel, expand it to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.model.prepare_tokens(x)

        for blk in self.model.blocks:
            x = blk(x)

        x = x[:, 1:] # remove the cls token

        return x
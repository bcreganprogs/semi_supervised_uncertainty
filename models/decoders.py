import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import SoftPositionEmbed
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
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
    """CNN Encoder to transform image to feature embeddings.	"""
    def __init__(self, slot_dim, num_channels=1):
        self.slot_dim = slot_dim
        self.num_channels = num_channels

        self.conv1 = ConvBlock(num_channels, 16) 
        self.conv2 = Downsample(16, 32)
        self.conv3 = Downsample(32, 64)
        self.conv4 = Downsample(64, 128)
        self.conv5 = Downsample(128, 256)
        self.conv6 = Downsample(256, 512)
        self.conv7 = Downsample(512, 512)



class Decoder(nn.Module):
    """Decoder to transform each slot to a 2D feature map."""
    def __init__(self, slot_dim, num_slots, num_classes, num_channels=1, initial_size=(8, 8)):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.num_channels = num_channels

        self.decoder_initial_size = initial_size
        self.decoder_pos_embeddings = SoftPositionEmbed(slot_dim, self.decoder_initial_size)
        
        self.conv1 = Upsample(slot_dim, 256)
        self.mid_downsample = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.conv2 = Upsample(256, 128)
        self.conv3 = Upsample(128, 128)
        self.conv4 = ConvBlock(128, 64)
        self.conv5 = Upsample(64, 32)
        self.conv6 = Upsample(32, 16)
        self.end_conv = nn.Conv2d(16, num_channels + 1, kernel_size=3, padding=1)

        self.final_conv = nn.Sequential(
            #nn.BatchNorm2d(num_slots),
            nn.Conv2d(num_slots, num_classes, kernel_size=1)
        )


    def forward(self, x):
        # input, x has shape (batch_size * num_slots, init_height, init_width, slot_dim)
   
        #x = self.decoder_pos_embeddings(x)   # (batch_size * num_slots, init_height, init_width, slot_dim)

        x = x.permute(0, 3, 1, 2)   # (batch_size * num_slots, slot_dim, init_height, init_width)
        #x = self.fc(x)
        x = self.conv1(x)   # (batch_size * num_slots, 512, 16, 16)
        x = self.mid_downsample(x)   # (batch_size * num_slots, 512, 14, 14)
        x = self.conv2(x)   # (batch_size * num_slots, 256, 28, 28)
        x = self.conv3(x)   # (batch_size * num_slots, 128, 56, 56)
        x = self.conv4(x)   # (batch_size * num_slots, 64, 112, 112)
        x = self.conv5(x)   # (batch_size * num_slots, 32, 224, 224)
        x = self.conv6(x)   # (batch_size * num_slots, 16, 224, 224)
        x = self.end_conv(x)   # (batch_size * num_slots, num_classes, 224, 224)
        x = F.relu(x)

        x = x.reshape(-1, self.num_slots,  self.num_channels + 1, 224, 224)   # (batch_size, num_slots, num_channels, 224, 224)

        #x = x.squeeze()   # (batch_size, num_slots, 224, 224)
       
        # if self.num_slots > self.num_classes:
        #     x = self.final_conv(x)

        return x
    
    
class SlotSpecificDecoder(nn.Module):
    """Decoder to transform each slot to a 2D feature map.
    This version has a separate decoder for each slot."""
    def __init__(self, slot_dim, num_slots, num_classes, include_recon=False, softmax_class=True, resolution=224,
                 image_chans=1, decoder_type='slot_specific'):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.resolution = resolution
        self.include_recon = include_recon
        self.softmax_class = softmax_class
        self.image_chans = image_chans
        self.decoder_type = decoder_type
        self.decoder_initial_size = (8, 8)
        self.decoder_pos_embeddings = SoftPositionEmbed(slot_dim, self.decoder_initial_size)

        if decoder_type == 'slot_specific':
            self.start_decoder = nn.ModuleList([self.make_start_decoder() for _ in range(num_slots)])
            self.end_decoder = nn.ModuleList([self.make_end_decoder() for _ in range(num_slots)])
        elif decoder_type == 'shared':
            self.start_decoder = self.make_start_decoder()
            self.end_decoder = self.make_end_decoder()

        # self.final_conv = nn.Sequential(
        #     nn.BatchNorm2d(num_slots),
        #     nn.Conv2d(num_slots*2, num_classes*2, kernel_size=1),
        #     nn.ReLU()
        # )

    def make_start_decoder(self):
        return nn.Sequential(
            ConvBlock(self.slot_dim, 512),   # (batch_size, 256, 8, 8)
            Upsample(512, 256),   # (batch_size, 256, 16, 16)
            # reduce from 16x16 to 14x14
            # nn.Conv2d(256, 256, kernel_size=3, padding=0),   # (batch_size, 256, 14, 14)
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            Upsample(256, 128),   # (batch_size, 128, 28, 28)
            Upsample(128, 64),   # (batch_size, 64, 56, 56)
            
        )
    
    def make_end_decoder(self):
        if not self.include_recon:
            return nn.Sequential(
                Upsample(64, 32),   # (batch_size, 32, 112, 112)
                Upsample(32, 16),    # (batch_size, 16, 224, 224)
                nn.Conv2d(16, self.image_chans, kernel_size=3, padding=1),   # (batch_size, 1, 224, 224)
            )
        else:
            return nn.Sequential(
                Upsample(64, 32),   # (batch_size, 16, 112, 112)
                Upsample(32, 16),    # (batch_size, 8, 224, 224)
                nn.Conv2d(16, self.image_chans + 1, kernel_size=3, padding=1),   # (batch_size, 1, 224, 224)
            )


    def forward(self, x):
        batch_size, num_slots, slot_dim, init_height, init_width = x.size()
        
        if self.decoder_type == 'slot_specific':
            # first slot decoding
            decoded_slots = []
            for i in range(self.num_slots):
                slot = x[:, i, :, :, :].squeeze(1)
                decoded = self.start_decoder[i](slot)
                decoded = self.end_decoder[i](decoded)
                decoded_slots.append(decoded)

            x = torch.stack(decoded_slots, dim=1).squeeze(2)  # (batch_size, num_slots, chans + 1, 224, 224)

        elif self.decoder_type == 'shared':
            x = self.start_decoder(x.view(-1, slot_dim, init_height, init_width))
            x = self.end_decoder(x)

            x = x[:, :, :self.resolution, :self.resolution]
        
        # if self.num_slots > self.num_classes and not self.include_recon:
        #     x = x.view(batch_size, -1, self.resolution, self.resolution)
        #     x = self.final_conv(x)
        #     x = x.view(batch_size, self.num_classes, -1, self.resolution, self.resolution)

        if not self.include_recon:
            x = x.squeeze()   # (batch_size, num_classes, self.resolution, self.resolution)
            x = x.view(batch_size, self.num_classes, self.resolution, self.resolution)
        else:
            x = x.view(batch_size, self.num_slots, self.image_chans + 1, self.resolution, self.resolution)

        return x
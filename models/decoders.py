import torch
import torch.nn as nn
import torch.nn.functional as F

# def build_grid(resolution):
#     ranges = [np.linspace(0., 1., num=res) for res in resolution]
#     grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
#     grid = np.stack(grid, axis=-1)
#     grid = np.reshape(grid, [resolution[0], resolution[1], -1])
#     grid = np.expand_dims(grid, axis=0)
#     grid = grid.astype(np.float32)
#     return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to('cuda:0')
    
# class SoftPositionEmbed(nn.Module):
#     def __init__(self, hidden_size, resolution):
#         """Builds the soft position embedding layer.
#         Args:
#         hidden_size: Size of input feature dimension.
#         resolution: Tuple of integers specifying width and height of grid.
#         """
#         super().__init__()
#         self.embedding = nn.Linear(4, hidden_size, bias=True)
#         self.grid = build_grid(resolution)

#     def forward(self, inputs):
#         grid = self.embedding(self.grid)
#         return inputs + grid

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.upsample_y = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x):

        # x is from last layer y is skip connection

        x = self.upsample(x)
        # y = self.upsample_y(y)

        # diffY = y.size()[2] - x.size()[2]
        # diffX = y.size()[3] - x.size()[3]

        # x = F.pad(x, (diffX // 2, diffX - diffX // 2,
        #               diffY // 2, diffY - diffY // 2))
        # x = torch.cat([y, x], dim=1)
        x = self.conv(x)

        return x

class Decoder(nn.Module):
    """Decoder to transform each slot to a 2D feature map."""
    def __init__(self, slot_dim, num_slots):
        super().__init__()
        self.slot_dim = slot_dim
        #self.num_classes = num_classes
        
        self.fc = nn.Linear(self.slot_dim, self.slot_dim * 4 * 4)
        self.conv1 = Upsample(self.slot_dim, 512)
        self.conv2 = Upsample(512, 256)
        self.conv3 = Upsample(256, 128)
        self.conv4 = Upsample(128, 64)
        self.conv5 = Upsample(64, 32)
        self.conv6 = Upsample(32, 16)
        self.conv7 = nn.Conv2d(16, 1, kernel_size=33, padding=0)

    def forward(self, x):
        # input, x has shape (batch_size, num_slots, slot_dim)
        batch_size, num_slots, slot_dim = x.size()

        x = self.fc(x)
        x = x.view(batch_size * num_slots, self.slot_dim, 4, 4)
        x = self.conv1(x)   # (batch_size * num_slots, 512, 8, 8)
        x = self.conv2(x)   # (batch_size * num_slots, 256, 16, 16)
        x = self.conv3(x)   # (batch_size * num_slots, 128, 32, 32)
        x = self.conv4(x)   # (batch_size * num_slots, 64, 64, 64)
        x = self.conv5(x)   # (batch_size * num_slots, 32, 128, 128)
        x = self.conv6(x)   # (batch_size * num_slots, 16, 256, 256)
        x = self.conv7(x)   # (batch_size * num_slots, 1, 224, 224)

        x = x.view(batch_size, num_slots, 224, 224)

        return x

class SlotReshape(nn.Module):
    def __init__(self, slot_dim, num_slots):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

    def forward(self, x):
        batch_size, _ = x.shape
        return x.view(batch_size, self.slot_dim, 4, 4)
    
    
class SlotSpecificDecoder(nn.Module):
    """Decoder to transform each slot to a 2D feature map.
    This version has a separate decoder for each slot."""
    def __init__(self, slot_dim, num_slots):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        #self.num_classes = num_classes

        self.decoders = nn.ModuleList([self.make_decoder() for _ in range(num_slots)])

    def make_decoder(self):
        return nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim * 4 * 4),
            SlotReshape(self.slot_dim, self.num_slots),
            Upsample(self.slot_dim, 512),
            Upsample(512, 256),
            Upsample(256, 128),
            Upsample(128, 64),
            Upsample(64, 32),
            Upsample(32, 16),
            nn.Conv2d(16, 1, kernel_size=33, padding=0)
        )

    def forward(self, x):
        # input, x has shape (batch_size, num_slots, slot_dim)
        batch_size, num_slots, slot_dim = x.size()
        decoded_slots = []
        for i in range(num_slots):
            decoded = self.decoders[i](x[:, i, :])  # (batch_size, 1, 224, 224)
            decoded_slots.append(decoded)

        x = torch.stack(decoded_slots, dim=1)   # (batch_size, num_slots, 1, 224, 224)
        x = x.view(batch_size, num_slots, 224, 224)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    batch_size, num_slots, slot_dim = slots.shape

    # Reshape and add spatial dimensions.
    slots = slots.view(-1, 1, 1, slot_dim)

    # # Broadcast to the resolution.
    grid = slots.expand(-1, resolution[0], resolution[1], -1)

    # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
    return grid

def unstack_and_split(x):
    """Separate masks and slot reconstructions. """
    
    recons = x[:, :, :-1, :, :]
    masks = x[:, :, -1:, :, :]
    return recons, masks


# def build_grid(resolution):
#     ranges = [np.linspace(0., 1., num=res) for res in resolution]
#     grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
#     grid = np.stack(grid, axis=-1)
#     grid = np.reshape(grid, [resolution[0], resolution[1], -1])
#     grid = np.expand_dims(grid, axis=0)
#     grid = grid.astype(np.float32)
#     return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


# class SoftPositionEmbed(nn.Module):
#     """Adds soft positional embedding with learnable projection."""

#     def __init__(self, hidden_size, resolution):
#         """Builds the soft position embedding layer.

#         Args:
#         hidden_size: Size of input feature dimension.
#         resolution: Tuple of integers specifying width and height of grid.
#         """
#         super().__init__()
#         self.fc = nn.Linear(4, hidden_size)
#         self.grid = self.build_grid(resolution)

#     def forward(self, inputs):
#         # inputs shape (batch_size*num_slots, width_init, height_init, slot_dim)
#         return inputs + self.fc(self.grid.to(inputs.device))

#     def build_grid(self, resolution):
#         ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
#         grid = torch.meshgrid(*ranges)
#         grid = torch.stack(grid, dim=-1)
#         grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
#         grid = grid.unsqueeze(0)
#         return torch.cat([grid, 1.0 - grid], dim=-1)

class SoftPositionEmbed(nn.Module):
    def __init__(self, out_channels: int, resolution):
        super().__init__()
        # (1, height, width, 4)
        self.register_buffer("grid", self.build_grid(resolution))
        self.mlp = nn.Linear(4, out_channels, bias=True)  # 4 for (x, y, 1-x, 1-y)

    def forward(self, x):
        # (1, height, width, out_channels)
        grid = self.mlp(self.grid)
        # (batch_size, out_channels, height, width)
        return x + grid     #.permute(0, 3, 1, 2)

    def build_grid(self, resolution):
        xy = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        xx, yy = torch.meshgrid(xy, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        grid = grid.unsqueeze(0)
        return torch.cat([grid, 1.0 - grid], dim=-1)
    
class ConvCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x1, x2):
        # Linear projections for queries, keys, and values
        queries = self.query_proj(x1)   # (batch_size, out_channels, 32, 32)
        keys = self.key_proj(x2)        # (batch_size, out_channels, 32, 32)
        values = self.value_proj(x2)    # (batch_size, out_channels, 32, 32)

        # Flatten the spatial dimensions for attention calculation
        b, c, h, w = queries.size()
        queries = queries.view(b, c, -1).permute(0, 2, 1)  # (batch_size, H*W, out_channels)
        keys = keys.view(b, c, -1)                         # (batch_size, out_channels, H*W)
        values = values.view(b, c, -1).permute(0, 2, 1)    # (batch_size, H*W, out_channels)

        # Compute attention scores and apply softmax
        attention_scores = torch.bmm(queries, keys) / (self.out_channels ** 0.5)  # (batch_size, H*W, H*W)
        attention_weights = F.softmax(attention_scores, dim=-1)                  # (batch_size, H*W, H*W)

        # Weighted sum of values
        out = torch.bmm(attention_weights, values)  # (batch_size, H*W, out_channels)

        # Reshape back to the original spatial dimensions
        out = out.permute(0, 2, 1).view(b, c, h, w)  # (batch_size, out_channels, 32, 32)
        
        return out


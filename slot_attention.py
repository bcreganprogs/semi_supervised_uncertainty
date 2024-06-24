import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SlotAttention(torch.nn.Module):
    def __init__(self, num_slots: int, dim: int, num_iterations: int, hidden_dim: int = 256):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.dim = dim

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_keys = torch.nn.Linear(dim, dim)         # from inputs
        self.to_queries = torch.nn.Linear(dim, dim)      # from slots
        self.to_values = torch.nn.Linear(dim, dim)       # from inputs

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.layer_norm_inputs = nn.LayerNorm(dim)
        self.layer_norm_slots = nn.LayerNorm(dim)
        self.layer_norm_pre_ff = nn.LayerNorm(dim)
    
    def forward(self, embeddings: torch.Tensor):
        """Slot Attention """
        B, N, D = embeddings.shape
        # 1) initialise the slots randomly
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_sigma.expand(B, self.num_slots, -1)
        slots = torch.normal(mu, sigma)

        embeddings = self.layer_norm_inputs(embeddings)

        keys, values = self.to_keys(embeddings), self.to_values(embeddings)

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.layer_norm_slots(slots)

            # 2) generate q, k and v vectors using linear projection
            #    keys and values are generated from the inputs and 
            #    queries are generated from the slots
            queries = self.to_queries(slots)

            # 3) calculate the attention weights between the slots and values
            dots = torch.einsum('bid,bjd->bij', queries, keys)
            attn = dots.softmax(-1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)        # scale attention
            
            # 4) calculate updated slot values by taking a weighted sum of the values
            slot_updates = torch.einsum('bjd,bij->bid', values, attn)

            # 5) GRU to update slots
            slots = self.gru(slot_updates.reshape(-1, D), slots_prev.reshape(-1, D))

            slots = slots.reshape(B, -1, D)
            slots = slots + self.fc2(F.relu(self.fc1(self.layer_norm_pre_ff(slots))))

        return slots


class ProbabalisticSlotAttention(torch.nn.Module):
    def __init__(self, num_slots: int, dim: int, num_iterations: int, hidden_dim: int = 256):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.dim = dim

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_keys = torch.nn.Linear(dim, dim)         # from inputs
        self.to_queries = torch.nn.Linear(dim, dim)      # from slots
        self.to_values = torch.nn.Linear(dim, dim)       # from inputs

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.layer_norm_inputs = nn.LayerNorm(dim)
        self.layer_norm_slots = nn.LayerNorm(dim)
        self.layer_norm_pre_ff = nn.LayerNorm(dim)
    
    def forward(self, embeddings: torch.Tensor):
        """Slot Attention """
        B, N, D = embeddings.shape
        # 1) initialise the slots randomly
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_sigma.expand(B, self.num_slots, -1)
        slots = torch.normal(mu, sigma)

        embeddings = self.layer_norm_inputs(embeddings)

        keys, values = self.to_keys(embeddings), self.to_values(embeddings)

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.layer_norm_slots(slots)

            # 2) generate q, k and v vectors using linear projection
            #    keys and values are generated from the inputs and 
            #    queries are generated from the slots
            queries = self.to_queries(slots)

            # 3) calculate the attention weights between the slots and values
            dots = torch.einsum('bid,bjd->bij', queries, keys)
            attn = dots.softmax(-1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)        # scale attention
            
            # 4) calculate updated slot values by taking a weighted sum of the values
            slot_updates = torch.einsum('bjd,bij->bid', values, attn)

            # 5) GRU to update slots
            slots = self.gru(slot_updates.reshape(-1, D), slots_prev.reshape(-1, D))

            slots = slots.reshape(B, -1, D)
            slots = slots + self.fc2(F.relu(self.fc1(self.layer_norm_pre_ff(slots))))

        return slots
    
def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to('cuda:0')
    
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


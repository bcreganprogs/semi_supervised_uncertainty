import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from scipy.optimize import linear_sum_assignment
import random

class SlotAttention(torch.nn.Module):
    def __init__(self, num_slots: int, dim: int, num_iterations: int, hidden_dim: int = 256, temperature: float = 1):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.dim = dim
        self.scale = dim ** -0.5
        self.temperature = temperature
        self.eps = 1e-8

        self.slots_mu = nn.Parameter(torch.zeros(1, 1, dim))
        #init.xavier_uniform_(self.slots_mu)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        #init.xavier_uniform_(self.slots_logsigma)

        self.to_keys = nn.Linear(dim, dim, bias=False)      # from inputs
        self.to_queries = nn.Linear(dim, dim, bias=False)   # from slots
        self.to_values = nn.Linear(dim, dim, bias=False)    # from inputs

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.layer_norm_inputs = nn.LayerNorm(dim)
        self.layer_norm_slots = nn.LayerNorm(dim)
        self.layer_norm_pre_ff = nn.LayerNorm(dim)
    
    def forward(self, embeddings: torch.Tensor):
        """Slot Attention """
        B, N, D = embeddings.shape
        # 1) initialise the slots 
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_logsigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn((B, self.num_slots, self.dim), device=embeddings.device)
        
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
            dots = torch.einsum('bkd,bnd->bkn', queries, keys) * self.scale     # shape (B, K, N)
            attn = F.softmax(dots / self.temperature, dim=1) + self.eps   # softmax along K
            attn_vis = attn
            attn = attn / attn.sum(dim=-1, keepdim=True)        # scale
            
            # inverse dot product attention
            # attn = 1 / (dots + 1e-8)
            # attn = attn / attn.sum(dim=-1, keepdim=True)        # scale
            
            # 4) calculate updated slot values by taking a weighted sum of the values
            slot_updates = torch.einsum('bnd,bkn->bkd', values, attn)

            # 5) GRU to update slots
            slots = self.gru(slot_updates.reshape(-1, D), slots_prev.reshape(-1, D))

            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.layer_norm_pre_ff(slots))

        return slots, attn_vis


class ProbabalisticSlotAttention(torch.nn.Module):
    """Implementation of Probabalistic Slot Attention from Identifiable Object-Centric Representation Learning
    via Probabilistic Slot Attention by Kori et al. 2024."""
    def __init__(self, num_slots: int, dim: int, num_iterations: int, hidden_dim: int = 256, temperature: float = 1):
        super(ProbabalisticSlotAttention, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.dim = dim
        self.temperature = temperature
        self.eps = 1e-8
        
        self.mixing_coeffs = nn.Parameter(1/self.num_slots * torch.ones(1, self.num_slots), requires_grad=False)  # shape (1, K)
        self.slots_mu = nn.Parameter(torch.randn(1, self.num_slots, dim))
        init.xavier_uniform_(self.slots_mu)
        #init.orthogonal_(self.slots_mu)
        self.slots_logsigma = nn.Parameter(torch.ones(1, self.num_slots, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_keys = torch.nn.Linear(dim, dim)         # from inputs
        self.to_queries = torch.nn.Linear(dim, dim)      # from slots
        self.to_values = torch.nn.Linear(dim, dim)       # from inputs

        self.layer_norm_inputs = nn.LayerNorm(dim)
        self.layer_norm_slots = nn.LayerNorm(dim)
        self.layer_norm_pre_ff = nn.LayerNorm(dim)
    
    def forward(self, embeddings: torch.Tensor):
        """Slot Attention """
        B, N, D = embeddings.shape
        # 1) initialise the slots randomly
        mixing_coeffs = self.mixing_coeffs.expand(B, -1)  # shape (B, K)
        mu = self.slots_mu.expand(B, -1, -1)  # shape (B, K, D)
        logsigma = self.slots_logsigma.expand(B, -1, -1)  # shape (B, K, D)
        sigma = logsigma.exp()
        slots = mu + sigma * torch.randn(mu.shape, device=embeddings.device) #* 0.1     # randomly initialise slots from standard normal shape (B, K, D)

        embeddings = self.layer_norm_inputs(embeddings)
        keys, values = self.to_keys(embeddings), self.to_values(embeddings)     # shape (B, N, D)

        for _ in range(self.num_iterations):
            # 2) generate queries from the slots
            queries = self.to_queries(slots)        # shape (B, K, D)
 
            # 3) calculate the attention weights by likelihood mixture of gaussians. Page 78 Pattern Recognition and Machine Learning by Bishop
            diff = keys.unsqueeze(2) - queries.unsqueeze(1)  # (B, N, 1, D), (B, 1, K, D) --> shape (B, N, K, D)
            gaussian_log_likelihood = -0.5 * torch.sum(diff**2 / (sigma.unsqueeze(1)**2 + self.eps), dim=-1)  # shape (B, N, K)
            gaussian_log_likelihood = -0.5 * D * np.log(2*np.pi) - 0.5 * torch.sum(torch.log(torch.abs(sigma) + self.eps), dim=-1).unsqueeze(1) + gaussian_log_likelihood  # shape (B, N, K)
            #gaussian_log_likelihood = torch.clip(gaussian_log_likelihood, -1000, 1000)
            # Compute attention weights
            attn = mixing_coeffs.unsqueeze(1) * gaussian_log_likelihood  # shape (B, N, K)
            attn = attn / attn.sum(dim=-1, keepdim=True)  # sum over slots
            attn = attn / (attn.sum(dim=1, keepdim=True) + self.eps)  # shape (B, N, K) sum over patches
     
            # Update slots
            mu = torch.einsum('bnk,bnd->bkd', attn, values)  # shape (B, K, D)
            
            # Update sigma
            diff = values.unsqueeze(2) - mu.unsqueeze(1)  # shape (B, N, K, D)
            sigma = torch.einsum('bnk,bnkd->bkd', attn, diff**2)
            mixing_coeffs = attn.sum(dim=1) / N     # sum along patch dimension and divide by number of patches
    
        # sample slots from mu and sigma
        sigma = torch.abs(sigma)  # ensure sigma is positive
        sigma = torch.clamp(sigma, min=self.eps)
        slots = mu + sigma * torch.randn_like(mu)  # shape (B, K, D)

        return slots, attn.permute(0, 2, 1)

class FixedSlotAttention(torch.nn.Module):
    def __init__(self, num_slots: int, dim: int, num_iterations: int, hidden_dim: int = 256, temperature: float = 1):
        super(FixedSlotAttention, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.dim = dim
        self.scale = dim ** -0.5
        self.temperature = temperature
        self.eps = 1e-8

        self.slots_mu = nn.Parameter(torch.zeros(1, self.num_slots, dim))
        #init.xavier_uniform_(self.slots_mu)
        #init.orthogonal_(self.slots_mu)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, self.num_slots, dim))
        #init.xavier_uniform_(self.slots_logsigma)

        self.to_keys = nn.Linear(dim, dim, bias=False)      # from inputs
        self.to_queries = nn.Linear(dim, dim, bias=False)   # from slots
        self.to_values = nn.Linear(dim, dim, bias=False)    # from inputs

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.layer_norm_inputs = nn.LayerNorm(dim)
        self.layer_norm_slots = nn.LayerNorm(dim)
        self.layer_norm_pre_ff = nn.LayerNorm(dim)
    
    def forward(self, embeddings: torch.Tensor):
        """Slot Attention """
        B, N, D = embeddings.shape
        # 1) initialise the slots 
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_logsigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=embeddings.device)
        
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
            # dots = torch.einsum('bid,bjd->bij', queries, keys) * self.scale     # shape (B, K, N)
            # attn = F.softmax((dots) / self.temperature, dim=1)   # softmax along K
            # attn_vis = attn
            #attn = attn / attn.sum(dim=-1, keepdim=True)        # scale
            
            # inverse dot product attention
            # attn = 1 / (dots + self.eps)
            # attn = attn / attn.sum(dim=-1, keepdim=True)        # scale

            agreement = torch.einsum("bkd,bdn->bkn", queries, keys.transpose(-2, -1))
            attn = agreement.softmax(dim=1) + 1e-8
            attn_vis = attn
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean
            # (b, k, d)
            slot_updates = torch.einsum("bkn,bnd->bkd", attn, values)
            
            # 4) calculate updated slot values by taking a weighted sum of the values
            #slot_updates = torch.einsum('bjd,bij->bid', values, attn)

            # 5) GRU to update slots
            slots = self.gru(slot_updates.reshape(-1, D), slots_prev.reshape(-1, D))

            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.layer_norm_pre_ff(slots))

        return slots, attn_vis


class FixedSlotAttentionMultiHead(torch.nn.Module):
    def __init__(self, num_slots: int, dim: int, num_iterations: int, hidden_dim: int = 256, temperature: float = 1, probabalistic: bool = False):
        super(FixedSlotAttentionMultiHead, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.dim = dim
        self.scale = dim ** -0.5
        self.temperature = temperature
        self.probabalistic = probabalistic
        self.eps = 1e-8

        self.slots_mu = nn.Parameter(torch.zeros(1, self.num_slots, dim))
        init.xavier_uniform_(self.slots_mu)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, self.num_slots, dim))
        init.xavier_uniform_(self.slots_logsigma)

        # 
        self.to_keys = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(self.num_slots)])      # from inputs
        self.to_queries = nn.Linear(dim, dim, bias=False)   # from slots
        self.to_values = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(self.num_slots)])    # from inputs

        if probabalistic:
            self.get_logsigma = nn.ModuleList([self.make_sigma_net() for _ in range(self.num_slots)])    # slot specific variance

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        self.layer_norm_inputs = nn.LayerNorm(dim)
        self.layer_norm_slots = nn.LayerNorm(dim)
        self.layer_norm_pre_ff = nn.LayerNorm(dim)

    def make_sigma_net(self):
        return nn.Sequential(
            nn.Linear(self.dim, self.dim*2, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim*2, self.dim),
            #nn.Softplus()      # not needed as is used to calculate logsigma
        )

    def make_mlp(self):
        return nn.Sequential(
            nn.Linear(self.dim, self.dim*2),
            nn.ReLU(),
            nn.Linear(self.dim*2, self.dim)
        )
    
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
    def forward(self, embeddings: torch.Tensor):
        """Slot Attention """
        B, N, D = embeddings.shape
        # 1) initialise the slots 
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_logsigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=embeddings.device)
        
        embeddings = self.layer_norm_inputs(embeddings)
        keys = [self.to_keys[i](embeddings) for i in range(self.num_slots)]
        values = [self.to_values[i](embeddings) for i in range(self.num_slots)]
        
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.layer_norm_slots(slots)

            # 2) generate q, k and v vectors using linear projection
            #    keys and values are generated from the inputs and 
            #    queries are generated from the slots
            queries = self.to_queries(slots)    # shape (B, N, D)
        
            # 3) calculate the attention weights between the slots and values
            dot_list = []
            for i in range(self.num_slots):
                dots = torch.einsum('bd,bnd->bn', queries[:, i], keys[i]) * self.scale   # shape (B, N)
                dot_list.append(dots)
            
            attn = torch.stack(dot_list, dim=1)   # shape (B, K, N)
            attn = F.softmax(attn / self.temperature, dim=1)   # softmax along K
            attn_vis = attn
            attn = attn / attn.sum(dim=-1, keepdim=True)        # scale

            # 4) calculate updated slot values by taking a weighted sum of the values
            for i in range(self.num_slots):
                slot_updates = torch.einsum('bnd,bn->bd', values[i], attn[:, i])
                slots[:, i] = self.gru(slot_updates, slots_prev[:, i])
            #slot_updates = torch.einsum('bjd,bij->bid', values, attn)

            # 5) GRU to update slots
            #slots = self.gru(slot_updates.reshape(-1, D), slots_prev.reshape(-1, D))

            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.layer_norm_pre_ff(slots)) # shape (B, K, D)

        if self.probabalistic:
            logsigmas = [self.get_logsigma[i](embeddings) for i in range(self.num_slots)]      # each of shape (B, N, 1)
            logsigmas = torch.stack(logsigmas, dim=1)       # shape (B, K, N, 1)
            #print(logsigmas.shape)      # shape (B, K, N, D)
            # slots shape (B, K, D)
            sigmas = logsigmas.exp().mean(dim=2) # shape (B, K, D)  average across patch dimension
            #kl_loss = self.kl_divergence(slots, sigmas)

        return slots, attn_vis#, kl_loss
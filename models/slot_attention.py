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
        self.slots_logsigma = nn.Parameter(torch.ones(1, self.num_slots, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_keys = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(self.num_slots)])      # from inputs
        self.to_queries = nn.Linear(dim, dim, bias=False)   # from slots
        self.to_values = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(self.num_slots)])    # from inputs

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
        slots = mu + sigma * torch.randn(mu.shape, device=embeddings.device)     # randomly initialise slots from standard normal shape (B, K, D)

        embeddings = self.layer_norm_inputs(embeddings)
        keys, values = self.to_keys(embeddings), self.to_values(embeddings)     # shape (B, N, D)

        for _ in range(self.num_iterations):
            # 2) generate queries from the slots
            queries = self.to_queries(slots)        # shape (B, K, D)
 
            # 3) calculate the attention weights by likelihood mixture of gaussians. Page 78 Pattern Recognition and Machine Learning by Bishop
            diff = keys.unsqueeze(2) - queries.unsqueeze(1)  # (B, N, 1, D), (B, 1, K, D) --> shape (B, N, K, D)
            gaussian_log_likelihood = -0.5 * torch.sum(diff**2 / (sigma.unsqueeze(1)**2 + self.eps), dim=-1)  # shape (B, K, N)
            gaussian_log_likelihood = -0.5 * D * np.log(2*np.pi) - 0.5 * torch.sum(torch.log(torch.abs(sigma) + self.eps), dim=-1).unsqueeze(1) + gaussian_log_likelihood  # shape (B, K, N)
            gaussian_log_likelihood = torch.clip(gaussian_log_likelihood, -10000, 10000)
            # Compute attention weights
            log_attn = torch.log(mixing_coeffs.unsqueeze(2) + self.eps) + gaussian_log_likelihood
            attn = F.softmax(log_attn, dim=1)
     
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
    def __init__(self, num_slots: int, dim: int, input_dim: int, num_iterations: int, num_heads: int = 1, hidden_dim: int = 1024, temperature: float = 1, posterior_sample: bool = False):
        super(FixedSlotAttentionMultiHead, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_heads = num_heads
        self.dim = dim
        self.scale = dim ** -0.5
        self.temperature = temperature
        self.posterior_sample = posterior_sample
        self.eps = 1e-8

        self.slots_mu = nn.Parameter(torch.zeros(1, self.num_slots, dim))
        init.xavier_uniform_(self.slots_mu)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, self.num_slots, dim))
        init.xavier_uniform_(self.slots_logsigma)

        # 
        self.to_keys = nn.Parameter(torch.rand(self.num_slots, input_dim, self.dim))      # from inputs
        self.to_queries = nn.Parameter(torch.rand(self.num_slots, input_dim, self.dim))   # from slots
        self.to_values = nn.Parameter(torch.rand(self.num_slots, self.dim, self.dim))    # from inputs

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        self.layer_norm_inputs = nn.LayerNorm(input_dim)
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
        keys = torch.einsum("bne,ked->bknd", embeddings, self.to_keys).view(B, self.num_slots, N, self.num_heads, self.dim // self.num_heads)
        values = torch.einsum("bne,ked->bknd", embeddings, self.to_values).view(B, self.num_slots, N, self.num_heads, self.dim // self.num_heads)
        
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.layer_norm_slots(slots)

            # 2) generate q, k and v vectors using linear projection
            #    keys and values are generated from the inputs and 
            #    queries are generated from the slots
            queries = torch.einsum('bkd,kdd->bkd', slots, self.to_queries).view(B, self.num_slots, self.num_heads, self.dim // self.num_heads)    # shape (B, K, H, D/H)
        
            # 3) calculate the attention weights between the slots and values
            dots = torch.einsum('bkhu,bknhu->bknh', queries, keys) * self.scale
            attn = F.softmax(dots / self.temperature, dim=1)   # softmax along K
            attn = attn.sum(dim=-1) / self.num_heads
            attn_vis = attn
            attn = attn / attn.sum(dim=-1, keepdim=True)        # scale

            # 4) calculate updated slot values by taking a weighted sum of the values
            slot_updates = torch.einsum('bknhu,bkn->bkhu', values, attn)
            slot_updates = slot_updates.view(B, self.num_slots, self.dim)

            # 5) GRU to update slots
            slots = self.gru(slot_updates.reshape(-1, D), slots_prev.reshape(-1, D))

            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.layer_norm_pre_ff(slots)) # shape (B, K, D)

        return slots, attn_vis
    
# class FixedSlotAttentionMultiHeadProb(torch.nn.Module):
#     """Implementation of Probabalistic Slot Attention from Identifiable Object-Centric Representation Learning
#     via Probabilistic Slot Attention by Kori et al. 2024."""
#     def __init__(self, num_slots: int, dim: int, num_iterations: int, num_heads: int = 1, hidden_dim: int = 256, temperature: float = 1, probabalistic: bool = False):
#         super(FixedSlotAttentionMultiHeadProb, self).__init__()
#         self.num_slots = num_slots
#         self.num_iterations = num_iterations
#         self.dim = dim
#         self.num_heads = num_heads
#         self.scale = dim ** -0.5
#         self.temperature = temperature
#         self.probabalistic = probabalistic
#         self.eps = 1e-5

#         self.slots_mu = nn.Parameter(torch.zeros(1, self.num_slots, dim))
#         init.xavier_uniform_(self.slots_mu)
#         self.slots_logsigma = nn.Parameter(torch.zeros(1, self.num_slots, dim))
#         init.xavier_uniform_(self.slots_logsigma)

#         # learnable weights
#         self.mixing_coeffs = nn.Parameter(1/self.num_slots * torch.ones(1, self.num_slots), requires_grad=False)  # shape (1, K)
#         self.to_keys = nn.Parameter(torch.rand(self.num_slots, self.dim, self.dim))      # from inputs
#         self.to_queries = nn.Parameter(torch.rand(self.num_slots, self.dim, self.dim))   # from slots
#         self.to_values = nn.Parameter(torch.rand(self.num_slots, self.dim, self.dim))    # from inputs

#         self.gru = nn.GRUCell(dim, dim)

#         hidden_dim = max(dim, hidden_dim)

#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim)
#         )

#         self.layer_norm_inputs = nn.LayerNorm(dim)
#         self.layer_norm_slots = nn.LayerNorm(dim)
#         self.layer_norm_pre_ff = nn.LayerNorm(dim)
    
#     def forward(self, embeddings: torch.Tensor):
#         """Slot Attention """
#         B, N, D = embeddings.shape
#         # 1) initialise the slots 
#         mu = self.slots_mu.expand(B, -1, -1)
#         sigma = self.slots_logsigma.exp().expand(B, -1, -1)
#         slots = mu + sigma * torch.randn(mu.shape, device=embeddings.device)
#         mixing_coeffs = self.mixing_coeffs.expand(B, -1).unsqueeze(2)  # shape (B, K, 1)
        
#         embeddings = self.layer_norm_inputs(embeddings)
#         keys = torch.einsum("bne,ked->bknd", embeddings, self.to_keys).view(B, self.num_slots, N, self.num_heads, self.dim // self.num_heads)
#         values = torch.einsum("bne,ked->bknd", embeddings, self.to_values).view(B, self.num_slots, N, self.num_heads, self.dim // self.num_heads)
        
#         for _ in range(self.num_iterations):
#             slots_prev = slots
#             slots = self.layer_norm_slots(slots) # shape (B, K, D)
            
#             # attention = mixture coefficients * likelihood of gaussian / sum of mixture coefficients * likelihood of gaussian
#             # Bishop Pattern Recognition and Machine Learning page 78
#             # find likelihood of keys under normal given by queries and sigma
#             queries = torch.einsum('bkd,kdd->bkd', slots, self.to_queries).view(B, self.num_slots, self.num_heads, self.dim // self.num_heads)    # shape (B, K, H, D/H)
    	    
#             exponent = -0.5 * (keys - queries.unsqueeze(2))**2 / (sigma.unsqueeze(2)**2 + self.eps)     # shape (B, K, N)
#             log_pi = -0.5 * D * torch.log(torch.tensor(2*torch.pi))                                     # shape (1)
#             log_scale = - torch.log(torch.clamp(sigma, min=self.eps)).unsqueeze(2)                      # shape (B, K, 1, D)
          
#             gaussian_log_likelihood = torch.log(mixing_coeffs + self.eps) + (log_pi + log_scale + exponent).sum(dim=-1)  # shape (B, K, N)
      
#             attn = F.softmax(gaussian_log_likelihood, dim=1) + self.eps                                 # shape (B, K, N)
       
#             attn_vis = attn

#             Nk = attn.sum(dim=2, keepdim=True)  # shape (B, K, 1)
#             slot_updates = (1 / Nk + self.eps) * torch.sum(attn.unsqueeze(-1) * values, dim=2)
#             #slot_updates = torch.einsum('bknd,bkn->bkd', values, attn)  # shape (B, K, D)
            
#             #var = torch.einsum('bknd,bkn->bkd', (values - slot_updates.unsqueeze(2))**2, attn)  # shape (B, K, D)
#             #sigma = torch.sqrt(var)
#             new_queries = torch.einsum('bkd,kdd->bkd', slot_updates, self.to_queries)
#             sigma = (1 / Nk) * torch.sum(attn.unsqueeze(-1) * (keys - new_queries.unsqueeze(2))**2, dim=2)  # shape (B, K, D)
#             sigma = torch.sqrt(sigma) + self.eps  # shape (B, K, D)
            
#             mixing_coeffs = Nk / N  # update mixing coefficients
            
#             # 5) GRU to update slots
#             slots = self.gru(slot_updates.reshape(-1, D), slots_prev.reshape(-1, D))

#             slots = slots.reshape(B, -1, D)
#             slots = slots + self.mlp(self.layer_norm_pre_ff(slots)) # shape (B, K, D)
            
#         # sample from slotwise distributions
#         slots = slots + sigma * torch.randn_like(slots)
        
#         return slots, attn_vis

# class FixedSlotAttentionMultiHeadProb(torch.nn.Module):
#     """Implementation of Probabalistic Slot Attention from Identifiable Object-Centric Representation Learning
#     via Probabilistic Slot Attention by Kori et al. 2024."""
#     def __init__(self, num_slots: int, dim: int, num_iterations: int, num_heads: int = 4, hidden_dim: int = 256, temperature: float = 1, probabalistic: bool = False):
#         super(FixedSlotAttentionMultiHeadProb, self).__init__()
#         self.num_slots = num_slots
#         self.num_iterations = num_iterations
#         self.dim = dim
#         self.num_heads = num_heads
#         self.scale = dim ** -0.5
#         self.temperature = temperature
#         self.probabalistic = probabalistic
#         self.eps = 1e-5

#         self.slots_mu = nn.Parameter(torch.zeros(1, self.num_slots, dim))
#         init.xavier_uniform_(self.slots_mu)
#         self.slots_logsigma = nn.Parameter(torch.zeros(1, self.num_slots, dim))
#         init.xavier_uniform_(self.slots_logsigma)

#         # learnable weights
#         self.mixing_coeffs = nn.Parameter(1/self.num_slots * torch.ones(1, self.num_slots), requires_grad=False)  # shape (1, K)
#         self.to_keys = nn.Parameter(torch.rand(self.num_slots, self.dim, self.dim))      # from inputs
#         self.to_queries = nn.Parameter(torch.rand(self.num_slots, self.dim, self.dim))   # from slots
#         self.to_values = nn.Parameter(torch.rand(self.num_slots, self.dim, self.dim))    # from inputs

#         self.gru = nn.GRUCell(dim, dim)

#         hidden_dim = max(dim, hidden_dim)

#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim)
#         )

#         self.layer_norm_inputs = nn.LayerNorm(dim)
#         self.layer_norm_slots = nn.LayerNorm(dim)
#         self.layer_norm_pre_ff = nn.LayerNorm(dim)
    
#     def forward(self, embeddings: torch.Tensor):
#         """Slot Attention """
#         B, N, D = embeddings.shape
#         # 1) initialise the slots 
#         mu = self.slots_mu.expand(B, -1, -1)
#         sigma = self.slots_logsigma.exp().expand(B, -1, -1)
#         slots = mu + sigma * torch.randn(mu.shape, device=embeddings.device)
#         mixing_coeffs = self.mixing_coeffs.expand(B, -1).unsqueeze(2)  # shape (B, K, 1)
        
#         embeddings = self.layer_norm_inputs(embeddings)
#         keys = torch.einsum("bne,ked->bknd", embeddings, self.to_keys).view(B, self.num_slots, N, self.num_heads, self.dim // self.num_heads) # shape (B, K, N, H, D/H)
#         values = torch.einsum("bne,ked->bknd", embeddings, self.to_values).view(B, self.num_slots, N, self.num_heads, self.dim // self.num_heads) # shape (B, K, N, H, D/H)
#         sigma = sigma.view(B, self.num_slots, self.num_heads, self.dim // self.num_heads)
        
#         for _ in range(self.num_iterations):
#             slots_prev = slots
#             slots = self.layer_norm_slots(slots) # shape (B, K, D)
            
#             # attention = mixture coefficients * likelihood of gaussian / sum of mixture coefficients * likelihood of gaussian
#             # Bishop Pattern Recognition and Machine Learning page 78
#             # find likelihood of keys under normal given by queries and sigma
#             queries = torch.einsum('bkd,kdd->bkd', slots, self.to_queries).view(B, self.num_slots, self.num_heads, self.dim // self.num_heads)    # shape (B, K, H, D/H)
#     	    # sigma shape B, K, D, query shape B, K, H, D/H, keys shape B, K, N, H, D/H
#             exponent = -0.5 * (keys - queries.unsqueeze(2))**2 / (sigma.unsqueeze(2)**2 + self.eps)     # shape (B, K, N)
#             log_pi = -0.5 * D * torch.log(torch.tensor(2*torch.pi))                                     # shape (1)
#             log_scale = - torch.log(torch.clamp(sigma, min=self.eps)).unsqueeze(2)                      # shape (B, K, 1, D)
       
#             gaussian_log_likelihood = torch.log(mixing_coeffs + self.eps).unsqueeze(-1) + (log_pi + log_scale + exponent).sum(dim=-1) * self.scale  # shape (B, K, N)
      
#             attn = F.softmax(gaussian_log_likelihood, dim=1) + self.eps                              # shape (B, K, N, H)
   
#             attn_vis = attn.sum(-1)

#             Nk = attn.sum(dim=2, keepdim=True)  # shape (B, K, 1, H)
#             Nk = Nk.permute(0, 1, 3, 2)  # shape (B, K, H, 1)
  
#             slot_updates = torch.einsum('bknh,bknhu->bkhu', attn, values)   # shape (B, K, H, D/H)
#             slot_updates = (1 / Nk + self.eps) * slot_updates  # shape (B, K, H, D/H)     

#             sigma = torch.sum(attn.unsqueeze(-1) * (values - slot_updates.unsqueeze(2))**2, dim=2) # shape (B, K, H, D/H)
#             sigma = (1 / Nk + self.eps) * sigma  # shape (B, K, D)
#             sigma = torch.sqrt(sigma) + self.eps # shape (B, K, D)
            
#             mixing_coeffs = Nk.sum(-2) / N  # update mixing coefficients
            
#             # 5) GRU to update slots
#             slots = self.gru(slot_updates.reshape(-1, D), slots_prev.reshape(-1, D))

#             slots = slots.reshape(B, self.num_slots, D)
#             slots = slots + self.mlp(self.layer_norm_pre_ff(slots)) # shape (B, K, D)
            
#         sigma = sigma.view(B, self.num_slots, self.dim)
#         # sample from slotwise distributions
#         slots = slots + sigma * torch.randn_like(slots)
        
#         return slots, attn_vis

class FixedSlotAttentionMultiHeadProb(torch.nn.Module):
    """Implementation of Probabalistic Slot Attention from Identifiable Object-Centric Representation Learning
    via Probabilistic Slot Attention by Kori et al. 2024."""
    def __init__(self, num_slots: int, slot_dim: int, input_dim: int, num_iterations: int, num_heads: int = 4, hidden_dim: int = 256, temperature: float = 1, posterior_sample: bool = False):
        super(FixedSlotAttentionMultiHeadProb, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_dim = slot_dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.scale = (slot_dim // num_heads) ** -0.5
        self.temperature = temperature
        self.posterior_sample = posterior_sample # whether to sample from the posterior or not
        self.eps = 1e-5   # need this large when training with MP16

        self.slots_mu = nn.Parameter(torch.zeros(1, self.num_slots, slot_dim))
        init.xavier_uniform_(self.slots_mu)
        # nn.init.orthogonal_(self.slots_mu)
        self.mu_scale = nn.Parameter(torch.zeros(1, self.num_slots, slot_dim))
        init.xavier_uniform_(self.mu_scale)
        self.slots_logsigma = nn.Parameter(torch.ones(1, self.num_heads, self.num_slots, slot_dim // num_heads))

        # learnable weights
        self.mixing_coeffs = nn.Parameter(1/self.num_slots * torch.ones(1, self.num_slots), requires_grad=False)  # shape (1, K)
        self.to_keys = nn.Parameter(torch.rand(self.input_dim, self.slot_dim))      # from inputs
        self.to_queries = nn.Parameter(torch.rand(self.num_slots, self.slot_dim, self.slot_dim))   # from slots
        self.to_values = nn.Parameter(torch.rand(self.input_dim, self.slot_dim))    # from inputs

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.layer_norm_inputs = nn.LayerNorm(input_dim)
        self.layer_norm_slots = nn.LayerNorm(slot_dim)
        self.layer_norm_pre_ff = nn.LayerNorm(slot_dim)
    
    def forward(self, embeddings: torch.Tensor):
        """Slot Attention """
        B, N, D = embeddings.shape
        # 1) initialise the slots 
        # if self.training:
        #     sigma = self.train_sigma.exp().expand(B, self.num_slots, -1)
        #     slots = self.train_mu + self.train_sigma.exp() * torch.randn(
        #         B, self.num_slots, self.slot_dim, device=embeddings.device
        #     )
        # else:
        mu = self.slots_mu.repeat(B, 1, 1)
        slots = mu + self.mu_scale.exp() * torch.randn(mu.shape, device=embeddings.device)

        sigma = self.slots_logsigma.repeat(B, 1, 1, 1)

        mixing_coeffs = self.mixing_coeffs.expand(B, -1).unsqueeze(2)  # shape (B, K, 1)
        
        embeddings = self.layer_norm_inputs(embeddings)
        keys = torch.einsum("bne,ed->bnd", embeddings, self.to_keys).view(B, N, self.num_heads, self.slot_dim // self.num_heads) # shape (B, N, H, D/H)
        values = torch.einsum("bne,ed->bnd", embeddings, self.to_values).view(B, N, self.num_heads, self.slot_dim // self.num_heads) # shape (B, K, N, H, D/H)
   
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = slots.detach()
            slots = self.layer_norm_slots(slots) # shape (B, K, D)

            # attention = mixture coefficients * likelihood of gaussian / sum of mixture coefficients * likelihood of gaussian
            # Bishop Pattern Recognition and Machine Learning page 78
            # find likelihood of keys under normal given by queries and sigma
            queries = torch.einsum('bkd,kdd->bkd', slots, self.to_queries).view(B, self.num_slots, self.num_heads, self.slot_dim // self.num_heads).unsqueeze(2)  # shape (B, K, H, 1, D/H)
            sigma = sigma.view(B, self.num_slots, self.num_heads, self.slot_dim // self.num_heads).unsqueeze(2) # shape (B, K, H, 1, D/H)

    	    # sigma shape B, K, D, query shape B, K, H, D/H, keys shape B, K, N, H, D/H
            exponent = -0.5 * (keys.unsqueeze(1) - queries)**2 / (sigma**2 + self.eps)     # shape (B, K, N)
            log_pi = -0.5 * D * torch.log(torch.tensor(2*torch.pi))                                     # shape (1)
            log_scale = - torch.log(torch.clamp(sigma, min=self.eps))                      # shape (B, K, 1, D)
 
            gaussian_log_likelihood = torch.log(mixing_coeffs + self.eps).unsqueeze(-1) + (log_pi + log_scale + exponent).sum(dim=-1) * self.scale  # shape (B, K, N)

            gaussian_log_likelihood = gaussian_log_likelihood.permute(0, 1, 3, 2).contiguous()  # shape (B, K, H, N)
            gaussian_log_likelihood = gaussian_log_likelihood.view(B, self.num_slots * self.num_heads, N)
            attn = F.softmax((gaussian_log_likelihood / self.temperature), dim=1) + self.eps                          # shape (B, K, N, H)
            attn = attn.view(B, self.num_slots, self.num_heads, N)
            attn = attn.permute(0, 1, 3, 2)  # shape (B, K, N, H)
     
            attn_vis = attn.sum(-1)

            Nk = attn.sum(dim=2, keepdim=True)  # shape (B, K, 1, H)
 
            Nk = Nk.permute(0, 1, 3, 2)  # shape (B, K, H, 1)
  
            slot_updates = torch.einsum('bknh,bnhu->bkhu', attn, values)   # shape (B, K, H, D/H)
            slot_updates = (1 / (Nk + self.eps)) * slot_updates  # shape (B, K, H, D/H)     

            sigma = torch.sum(attn.unsqueeze(-1) * (values.unsqueeze(1) - slot_updates.unsqueeze(2))**2, dim=2) # shape (B, K, H, D/H)
            sigma = (1 / (Nk + self.eps)) * sigma  # shape (B, K, D)
            sigma = torch.clamp(sigma, min=self.eps)
            sigma = torch.sqrt(sigma) + self.eps  # shape (B, K, D)

            mixing_coeffs = Nk.sum(-2) / N  # update mixing coefficients
            
            # 5) GRU to update slots
            slots = self.gru(slot_updates.view(-1, self.slot_dim), slots_prev.view(-1, self.slot_dim))

            slots = slots.reshape(B, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.layer_norm_pre_ff(slots)) # shape (B, K, D)
            
        sigma = sigma.view(B, self.num_slots, self.slot_dim)
        # sample from slotwise distributions
        if self.posterior_sample:
            slots = slots + sigma * torch.randn_like(slots)
        
        return slots, attn_vis, values.detach()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
from timm.models.vision_transformer import Block

from models.utils_spot import *

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    
    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, :T])

class TransformerDecoderBlock(nn.Module):
    
    def __init__(self, max_len, d_model, num_heads, dropout=0., gain=1., is_first=False, num_cross_heads=None):
        super().__init__()
        
        self.is_first = is_first
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)
        
        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        
        if num_cross_heads is None:
            num_cross_heads = num_heads
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_cross_heads, dropout, gain)
        
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))
    
    
    def forward(self, input, encoder_output, causal_mask=False):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = input.shape[1]

        self_attn_mask = self.self_attn_mask[:T, :T] if causal_mask else None
        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input, self_attn_mask)
            input = input + x     # combine with post layer norm
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, self_attn_mask)
            input = input + x         # combine with pre layer norm
        
        x = self.encoder_decoder_attn_layer_norm(input)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        input = input + x
        
        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerDecoder(nn.Module):
    
    def __init__(self, num_blocks, max_len, d_model, num_heads, dropout=0., num_cross_heads=None):
        super().__init__()
        
        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain, is_first=True)] +
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain, is_first=False, num_cross_heads=num_cross_heads)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    
    def forward(self, input, encoder_output, causal_mask=False):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            input = block(input, encoder_output, causal_mask)
        
        return self.layer_norm(input)

class TransformerDecoderImage(nn.Module):
    
    def __init__(self, num_blocks, max_len, d_model, num_heads, dropout=0., num_cross_heads=None, patch_size=16, out_chans=2,
                autoregressive=False):
        super().__init__()
        
        self.decoder_initial_size = (8, 8)
        self.autoregressive = autoregressive
        self.patch_size = patch_size

        if not autoregressive:
            self.out_chans = 1
        else:
            self.out_chans = out_chans

        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain, is_first=True)] +
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain, is_first=False, num_cross_heads=num_cross_heads)
                 for _ in range(num_blocks - 1)])	
        else:
            self.blocks = nn.ModuleList()

        self.layer_norm_recons = nn.LayerNorm(d_model)

        # self.layer_norm_decoder = nn.LayerNorm(d_model)

        # self.decoder_pos_embed = PositionalEncoding(max_len, d_model, dropout)

        # self.decoder_blocks = nn.ModuleList([
        #     Block(d_model, num_heads, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
        #     for i in range(num_blocks)])

        if self.autoregressive:
            self.recon_projs = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, patch_size**2 * 4, bias=True),
                        nn.LeakyReLU(),
                        nn.Linear(patch_size**2 * 4, patch_size**2 * 4, bias=True),
                        nn.LeakyReLU(),
                        nn.Linear(patch_size**2 * 4, patch_size**2, bias=True)
                    ) for _ in range(out_chans)
                ])
            self.mask_projs = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, patch_size**2 * 4, bias=True),
                        nn.LeakyReLU(),
                        nn.Linear(patch_size**2 * 4, patch_size**2 * 4, bias=True),
                        nn.LeakyReLU(),
                        nn.Linear(patch_size**2 * 4, patch_size**2, bias=True)
                    ) for _ in range(out_chans)
                ])
            # self.recon_projs = nn.ModuleList([nn.Linear(d_model, patch_size**2, bias=True) for _ in range(out_chans)])
            # self.mask_projs = nn.ModuleList([nn.Linear(d_model, patch_size**2, bias=True) for _ in range(out_chans)])
        else:
            self.recon_projs = nn.ModuleList([nn.Linear(d_model, patch_size**2, bias=True)])
            self.mask_projs = nn.ModuleList([nn.Linear(d_model, patch_size**2, bias=True)])

        self.recon_projs.apply(self.init_weights)
        self.mask_projs.apply(self.init_weights)    
        
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Parameter)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)  

    def forward(self, input, encoder_output, causal_mask=False, inv_perm_indices=None):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            input = block(input, encoder_output, causal_mask)

        input = self.layer_norm_recons(input)

        # input = input + self.decoder_pos_embed(input)

        # for blk in self.decoder_blocks:
        #     input = blk(input)

        # input = self.layer_norm_decoder(input)

        if inv_perm_indices is not None:
            input = input[:, inv_perm_indices, :]

        recons = []
        masks = []
    
        for slot in range(self.out_chans):
            # split last dim in 2, shape: (batch_size, num_patches, patch_size**2, 2)
            recon = self.recon_projs[slot](input)
            mask = self.mask_projs[slot](input)

            recons.append(recon)
            masks.append(mask)

        recons = torch.stack(recons, dim=1)
        masks = torch.stack(masks, dim=1) # shape (batch _size, num_slots, num_patches, patch_size**2)
       
        return recons, masks
    
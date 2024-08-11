import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

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
    
    
    def forward(self, input, encoder_output, causal_mask=True):
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
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, self_attn_mask)
            input = input + x
        
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
    
    
    def forward(self, input, encoder_output, causal_mask=True):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            input = block(input, encoder_output, causal_mask)
        
        return self.layer_norm(input)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from timm.models.vision_transformer import PatchEmbed, Block
from pytorch_lightning.strategies import DDPStrategy
import wandb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size[0], grid_size[1])
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        embed_dim=1024,
        num_channels=3,
        num_heads=16,
        depth=24,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_layer = nn.LayerNorm,
        mlp_ratio=4.0,
        patch_size=16,
        norm_pix_loss=False,
        mask_ratio=0.75,
        dropout=0.0,
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_classes: Number of classes to predict
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = num_channels
        self.mask_ratio = mask_ratio

        # -------ENCODER PART------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, num_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    attn_drop=dropout,
                    proj_drop=dropout
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # -------DECODER PART------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    attn_drop=dropout,
                    proj_drop=dropout
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * num_channels, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=self.patch_embed.grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            grid_size=self.patch_embed.grid_size,
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_patch_sequence(self, x):
        patch_height, patch_width = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]

        # unfold 2 extracts patches from height dimension while unfold 3 extracts patches from width dimension
        patches = x.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        # ensure patches are contiguous
        patches = patches.contiguous().view(-1, x.size(0), x.size(1), patch_height, patch_width)
        # permute to have shape (batch_size, num_patches, channels, patch_size, patch_size)
        patches = patches.permute(1, 0, 2, 3, 4)

        # create sequence of pixels
        n, c, h, w = ((x.size(0), x.size(1), x.size(2) // patch_height, x.size(3) // patch_width))

        x = x.reshape(shape=(n, c, h, patch_height, w, patch_width))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(n, h * w, patch_height* patch_height * c))

        return x

    def reverse_patch_sequence(self, x):
        # fold patches back to image
        c = self.in_channels
        p, q = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
        h, w = self.patch_embed.img_size[0], self.patch_embed.img_size[1]
        n = x.shape[0] # batch size
        x = x.reshape(shape=(n, h // p, w // q, p, q, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(n, c, h, w))

        return imgs

    def create_masks(self, patches):
        """mask specified proportion of patches"""
        device = patches.device

        batch_size, num_patches, _, = patches.shape
        num_masked_tokens = int(num_patches * self.mask_ratio)

        # Generate random values and sort to get mask indices
        random_values = torch.rand(batch_size, num_patches, device=device)
        indices = torch.argsort(random_values, dim=1)

        mask_indices = indices[:, :num_masked_tokens]

        # Create mask and scatter values to mask indices
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, mask_indices, True)

        # Clone patches and apply mask
        masked_patches = patches.clone()
        masked_patches[mask] = 0

        # Get reverse indices for potential reconstruction
        ids_reverse = torch.argsort(indices, dim=1)

        return masked_patches, mask, ids_reverse, num_masked_tokens

    def encoder(self, x):
        # Preprocess input

        x = self.patch_embed(x)

        # add positional embedding
        x = x + self.pos_embed[:, 1:, :]

        # perform random masking
        masked_image, mask, mask_indices, num_masked_tokens = self.create_masks(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(masked_image.shape[0], -1, -1)
  
        x = torch.cat((cls_tokens, masked_image), dim=1)

        # Apply Transforrmer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, mask_indices, num_masked_tokens
    
    def decoder(self, x, mask_indices, num_masked_tokens):
        device = x.device

        x = self.decoder_embed(x)
        
        # # add mask tokens    
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked_tokens, 1)

        # x[:, 1:, :] removes the first token (CLS token) from x
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1).to(device)  # no cls token
        mask_indices = mask_indices.to(device)
        x_ = torch.gather(x_, dim=1, index=mask_indices.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        target = self.create_patch_sequence(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, imgs):
        latent,  mask, mask_indices, num_masked_tokens = self.encoder(imgs)
        pred = self.decoder(latent, mask_indices, num_masked_tokens)  # [N, L, p*p*1]
        #print(pred.shape)
        loss = self.loss(imgs, pred, mask)
        # convert predicted patches into image
        pred = self.reverse_patch_sequence(pred)
        return loss, imgs, pred
    

class ViTAE(LightningModule):
    def __init__(self, model_kwargs, learning_rate: float = 0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        # define transformer block
        self.model = VisionTransformer(**model_kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, imgs, pred = self.model.forward(batch['image'])

        self.log("train_loss", loss, prog_bar=True)

        if batch_idx % 250 == 0:
            self._log_train_images(imgs, pred)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, imgs, pred = self.model.forward(batch['image'])

        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self._log_val_images(imgs[:5], pred[:5])


    def test_step(self, batch, batch_idx):
        pass

    def _log_val_images(self, imgs: torch.Tensor, preds: torch.Tensor):

        grid = make_grid(imgs[:5])
        grid_val = make_grid(preds[:5])
        self.logger.experiment.log({
                "Validation Images": [wandb.Image(grid, caption="Original Images"),
                           wandb.Image(grid_val, caption="Masked and Reconstructed Images")],
            })
        
    def _log_train_images(self, imgs: torch.Tensor, preds: torch.Tensor):

        grid = make_grid(imgs[:5])
        grid_val = make_grid(preds[:5])
        self.logger.experiment.log({
                "Train Images": [wandb.Image(grid, caption="Original Images"),
                           wandb.Image(grid_val, caption="Masked and Reconstructed Images")],
            })
        
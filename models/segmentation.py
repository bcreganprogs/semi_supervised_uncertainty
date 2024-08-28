import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from torchmetrics.functional import dice
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from pytorch_lightning import LightningModule
import wandb

from utils_local.utils import SoftPositionEmbed, spatial_broadcast, unstack_and_split
from models.slot_attention import ProbabalisticSlotAttention, FixedSlotAttention, SlotAttention, FixedSlotAttentionMultiHead, FixedSlotAttentionMultiHeadProb
from models.decoders import CNNDecoder
from models.transformers import TransformerDecoderImage

class ObjectSpecificSegmentation(LightningModule):
    def __init__(self, encoder, num_slots, num_iterations, num_classes, slot_dim=128, include_seg_loss=False, slot_attn_type='standard', probabilistic_sample: bool = True,
                decoder_type='cnn', 
                learning_rate=1e-3, image_chans=1, embedding_dim=768, image_resolution=224, temperature=1, freeze_encoder=False, lr_warmup=True, log_images=True,
                embedding_shape=None, include_pos_embed=False, patch_size=16, num_patches=196,
                slot_attn_heads=6, decoder_blocks=6, decoder_heads=6, decoder_dim=512, autoregressive=False, label_smoothing=0.0):
        super(ObjectSpecificSegmentation, self).__init__()

        self.encoder = encoder
        self.embedding_shape = (image_resolution // 2**4, image_resolution // 2**4) if embedding_shape is None else embedding_shape
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, embedding_shape[0]*embedding_shape[1], embedding_dim))
        nn.init.normal_(self.encoder_pos_embed, std=0.02)

        # slot attention
        if slot_attn_type == 'standard':
            self.slot_attention = SlotAttention(num_slots, slot_dim, num_iterations, temperature=temperature)
        elif slot_attn_type == 'probabilistic':
            self.slot_attention = FixedSlotAttentionMultiHeadProb(num_slots=num_slots, slot_dim=slot_dim, input_dim=embedding_dim, num_iterations=num_iterations, temperature=temperature, num_heads=slot_attn_heads,
            posterior_sample=probabilistic_sample)
        elif slot_attn_type == 'fixed':
            self.slot_attention = FixedSlotAttention(num_slots=num_slots, slot_dim=slot_dim, input_dim=embedding_dim, num_iterations=num_iterations, temperature=temperature)
        else:
            self.slot_attention = FixedSlotAttentionMultiHead(num_slots, slot_dim, input_dim=embedding_dim, num_iterations=num_iterations, temperature=temperature, num_heads=slot_attn_heads)
            
        # decoder
        self.decoder_type = decoder_type

        if decoder_type == 'transformer':
            self.patch_size = patch_size
            self.num_patches = num_patches
            self.decoder = TransformerDecoderImage(
                decoder_blocks, num_patches, decoder_dim, num_heads=decoder_heads, dropout=0.0, num_cross_heads=None, patch_size=patch_size, out_chans=num_slots,
                autoregressive=autoregressive)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            torch.nn.init.normal_(self.mask_token, std=0.02)
            self.input_proj = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim, bias=False),
                nn.LayerNorm(decoder_dim),
            )
            self.input_proj.apply(self.init_weights)
            self.slot_proj = nn.Sequential(
                nn.Linear(slot_dim, decoder_dim, bias=False),
                nn.LayerNorm(decoder_dim),
            )
            self.slot_proj.apply(self.init_weights)

            self.spatial_conv = nn.Sequential(
                nn.Conv2d(num_slots, num_slots * 8, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(num_slots * 8, num_slots, kernel_size=3, padding=1),
            )
            self.spatial_conv.apply(self.init_weights)

            self.autoregressive = autoregressive

            if autoregressive:
                self.permutations = [torch.arange((image_resolution // patch_size)**2)]
                self.bos_token = nn.Parameter(torch.zeros(1, 1, embedding_dim)) # no first dim as only using 1 permutation
                self.input_proj = nn.Sequential(
                    nn.Linear(embedding_dim, decoder_dim, bias=False),
                    nn.LayerNorm(decoder_dim),
                )
                # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
                torch.nn.init.normal_(self.bos_token, std=.02)
                # torch.nn.init.normal_(self.mask_token, std=.02)
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))
                torch.nn.init.normal_(self.pos_embed, std=0.02)
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
                torch.nn.init.normal_(self.pos_embed, std=0.02)

        else:
            self.decoder = CNNDecoder(decoder_dim, slot_dim, num_slots, num_classes, image_chans=image_chans, decoder_type='shared',
                                               resolution=image_resolution)
            self.decoder_pos_embeddings = nn.Parameter(torch.zeros(1, self.decoder.decoder_initial_size[0]*self.decoder.decoder_initial_size[1], slot_dim))

        # hyperparameters
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.image_chans = image_chans
        self.learning_rate = learning_rate
        self.lr_warmup = lr_warmup
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.include_seg_loss = include_seg_loss
        self.image_resolution = image_resolution
        self.include_pos_embed = include_pos_embed
        self.label_smoothing = label_smoothing
        self.freeze_encoder = freeze_encoder
  
        if freeze_encoder:
            encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

        self.decoder.apply(self.init_weights)
        self.slot_attention.apply(self.init_weights)

        self.test_predmaps = []
        self.test_probmaps = []

        # for logging
        self.log_images = log_images
        if log_images:
            self.save_hyperparameters(ignore=['encoder'])

        self.train_imgs = None
        self.train_preds = None
        self.train_attn = None
        self.train_masks = None
        self.train_recons = None
        self.train_preds = None
        self.val_imgs = None
        self.val_preds = None
        self.val_attn = None
        self.val_masks = None
        self.val_recons = None
        self.val_preds = None
        self.logged_train_images_this_epoch = False
        self.logged_val_images_this_epoch = False

    def forward(self, x):
        
        batch_size, num_channels, _, _ = x.size()
        # try:
            #patch_embeddings, _, _ = self.encoder.model.forward_encoder(x, 0.0)   # shape (batch_size, num_patches, embed_dim)
        encoder_type = type(self.encoder).__name__
        if self.freeze_encoder:
            with torch.no_grad():
                # if x.shape[-1] % self.patch_size != 0:
                #     x = nn.functional.pad(x, (0, self.patch_size - x.shape[-1] % self.patch_size, 0, self.patch_size - x.shape[-2] % self.patch_size), value=0)
                # x = nn.functional.pad(x, (3, 3, 3, 3), value=0)
                
                if encoder_type.startswith('Dino'):
                    patch_embeddings = self.encoder(x)
                elif encoder_type.startswith('ViTAE'):
                    patch_embeddings, _, _ = self.encoder.model.forward_encoder(x, 0.0)   # shape (batch_size, num_patches, embed_dim)
                    patch_embeddings = patch_embeddings[:, 1:, :]       # exclude cls token
                elif encoder_type.startswith('ResNet'):
                    patch_embeddings = self.encoder(x)
                    patch_embeddings = patch_embeddings.permute(0, 2, 3, 1) # shape (batch_size, 8, 8, embed_dim)
                else:
                    raise ValueError(f"Unsupported encoder type: {encoder_type}")
                
        else:
            if encoder_type.startswith('Dino'):
                patch_embeddings = self.encoder(x)
            elif encoder_type.startswith('ViTAE'):
                patch_embeddings, _, _ = self.encoder.model.forward_encoder(x, 0.0)   # shape (batch_size, num_patches, embed_dim)
                patch_embeddings = patch_embeddings[:, 1:, :]       # exclude cls token
            elif encoder_type.startswith('ResNet'):
                patch_embeddings = self.encoder(x)
                patch_embeddings = patch_embeddings.permute(0, 2, 3, 1) # shape (batch_size, 8, 8, embed_dim)
            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
            
        # add position embeddings
        if self.include_pos_embed:
            # reshape patch embeddings, flatten dim 1 and 
            patch_embeddings = patch_embeddings.view(batch_size, -1, patch_embeddings.shape[-1])  # shape (batch_size, num_patches, embed_dim)
            patch_embeddings = patch_embeddings + self.encoder_pos_embed.to(patch_embeddings.dtype)
        
        patch_embeddings = patch_embeddings.view(patch_embeddings.shape[0], -1, patch_embeddings.shape[-1])  # shape (batch_size, num_patches, embed_dim)

        slots, attn, sa_values = self.slot_attention(patch_embeddings)  # shape (batch_size, num_slots, slot_dim) sa_values is detached
        
        if self.decoder_type == 'cnn':
            # spatial broadcast
            x = slots.view(-1, self.slot_dim).unsqueeze(-1).unsqueeze(-1)
            h, w = self.decoder.decoder_initial_size
            x = x.expand(-1, -1, h, w)  # shape: (batch_size * num_slots, slot_dim, h, w)
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, h * w, self.slot_dim) # shape: (batch_size * num_slots, h, w, slot_dim)
            x = x + self.decoder_pos_embeddings.to(x.dtype)
            x = x.view(batch_size, self.num_slots, h, w, self.slot_dim)
            x = x.permute(0, 1, 4, 2, 3)
            x = self.decoder(x)  # shape: (batch_size * num_slots, image_chans + 1, 224, 224)
            decoded, masks = torch.split(x, [self.image_chans, 1], dim=2)  # shape: (batch_size * num_slots, image_chans, 224, 224), (batch_size * num_slots, 1, 224, 224)
                
        else:
            if self.autoregressive:
                bos_token = self.bos_token.expand(patch_embeddings.shape[0], -1, -1)
                # mask_token = self.mask_token.expand(patch_embeddings.shape[0], -1, -1)
                dec_input = torch.cat((bos_token, patch_embeddings[:, :-1, :]), dim=1)

                dec_input = dec_input + self.pos_embed.to(patch_embeddings.dtype)

                dec_input = self.input_proj(dec_input) # B, N, D
                slots = self.slot_proj(slots) # B, K, D

                decoded, masks = self.decoder(dec_input, slots, causal_mask=True)      # shape (batch_size, num_slots, patches, h * w)

                p = int(self.num_patches**0.5)  # number of patches in one dimension

                decoded = decoded.view(batch_size, self.num_slots, p, p, self.patch_size, self.patch_size)
                decoded = torch.einsum('nchwpq->nchpwq', decoded).contiguous()  # permute
                decoded = decoded.view(batch_size, self.num_slots, p * self.patch_size, p * self.patch_size).unsqueeze(2)

                masks = masks.view(batch_size * self.num_slots, p, p, self.patch_size, self.patch_size)
                masks = torch.einsum('nhwpq->nhpwq', masks).contiguous()
                masks = masks.view(-1, self.num_slots, p * self.patch_size, p * self.patch_size)

                # spatial convolution
                masks = self.spatial_conv(masks).unsqueeze(2)

            else:
                #dec_input = sa_values.view(batch_size, -1, self.slot_dim) # B, N, D 
                dec_input = self.mask_token.expand(batch_size * self.num_slots, -1, -1)
                # add position embeddings
                dec_input = dec_input + self.pos_embed.to(patch_embeddings.dtype)

                dec_input = self.input_proj(dec_input) # B, N, D
                
                slots = slots.view(-1, self.slot_dim).unsqueeze(1)                      # shape (batch_size * num_slots, 1, D)
                slots = self.slot_proj(slots)

                decoded, masks = self.decoder(dec_input, slots, causal_mask=False)      # shape (batch_size * num_slots, patches, h * w)
                
                p = int(self.num_patches**0.5)  # number of patches in one dimension

                decoded = decoded.view(batch_size * self.num_slots, p, p, self.patch_size, self.patch_size)
                decoded = torch.einsum('nhwpq->nhpwq', decoded).contiguous()
                decoded = decoded.view(-1, self.num_slots, p * self.patch_size, p * self.patch_size).unsqueeze(2)

                masks = masks.view(batch_size * self.num_slots, p, p, self.patch_size, self.patch_size)
                masks = torch.einsum('nhwpq->nhpwq', masks).contiguous()
                masks = masks.view(-1, self.num_slots, p * self.patch_size, p * self.patch_size)

                # spatial convolution
                masks = self.spatial_conv(masks).unsqueeze(2)
        
        logits = masks.clone()

        masks = F.softmax(masks, dim=1)         # softmax over slots

        # check if nan in masks and decoded
        recons = torch.sum(decoded * masks, dim=1)
        recons = torch.sigmoid(recons)

        return recons, masks, logits, attn  
    
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Parameter, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        
        if not self.lr_warmup:
            return optimizer

        scheduler = self.warmup_lr_scheduler(optimizer, 10000, 50000, 1e-6, self.learning_rate)
        return {
            "optimizer": optimizer,
             "lr_scheduler": {
                 "scheduler": scheduler,
            #     #"monitor": "train_loss",  # metric to monitor
                 "frequency": 1,  
                 "interval": "step",
            #     #"strict": True,
             },
        }

    # def configure_optimizers(self):
    #     # Define the parameters for Slot Attention with a specific learning rate
    #     slot_attention_params = {
    #         "params": self.slot_attention.parameters(),  # Adjust this to the correct attribute name
    #         "lr": 1e-5
    #     }

    #     # Define the parameters for the rest of the model with the standard learning rate
    #     other_params = {
    #         "params": [param for name, param in self.named_parameters() if "slot_attention" not in name],
    #         "lr": self.learning_rate
    #     }

    #     # Create the optimizer with the two parameter groups
    #     optimizer = torch.optim.Adam(
    #         [slot_attention_params, other_params]
    #     )

    #     return optimizer

    #     if not self.lr_warmup:
    #         return optimizer

    #     # Set up the learning rate scheduler if warmup is enabled
    #     scheduler = self.warmup_lr_scheduler(optimizer, 10000, 50000, 1e-6, self.learning_rate)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "frequency": 1,
    #             "interval": "step",
    #         },
    #     }

    
    def warmup_lr_scheduler(self, optimizer, warmup_steps, decay_steps, start_lr, target_lr):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps)) * (target_lr - start_lr) / target_lr + start_lr / target_lr
            else:
                return 0.5 ** ((current_step - warmup_steps) / decay_steps)
        return LambdaLR(optimizer, lr_lambda)

    def process_batch(self, batch, batch_idx):
        if self.include_seg_loss:
            x, y = batch['image'], batch['labelmap']
            # select random from second dimension
            y = y[:, torch.randint(0, y.shape[1], (1,))]

            if 'ground_truth' in batch.keys():
                gt = batch['ground_truth']
            else:
                gt = None
        else:
            x = batch['image']
        
        x = x.float() 

        recons, masks, logits, attn = self(x)
        
        mask_probs = logits[:, :self.num_classes]
        mask_probs[:, 0] = mask_probs[:, 0] + torch.sum(logits[:, self.num_classes:], dim=1)  # add background class to background mask

        preds = torch.argmax(masks, dim=1)
        preds[preds > (self.num_classes - 1)] = 0

        recon_loss = F.mse_loss(recons.squeeze(), x.squeeze()) 

        if self.include_seg_loss:
            loss = 0.0 * recon_loss
            # only consider masks and y for which y is not all zeros
            # indices = torch.where(y.squeeze().sum(dim=(1, 2)) > 0)[0]
            # loss_masks = masks[indices]
            # y = y[indices]

            # if  y is not empty
            # if len(indices) > 0:
            # mask_probs = mask_probs[:, 3:-3, 3:-3]

            # clip to avoid nan
            mask_probs = torch.clamp(mask_probs, -10, 10)
            seg_loss = F.cross_entropy(mask_probs, y, reduction='mean', label_smoothing=self.label_smoothing)#, ignore_index=0)     # segmentation loss

            loss += seg_loss 
            #preds = preds[indices]
            if gt is not None:
                dsc = dice(preds, gt, average='macro', num_classes=self.num_classes, ignore_index=0)
            else:
                dsc = dice(preds, y, average='macro', num_classes=self.num_classes, ignore_index=0)
        else:
            loss = recon_loss
            dsc = 0

        probs = torch.softmax(recons, dim=1)
            
        # else:
        #     x, y = batch['image'], batch['labelmap']
        #     logits, masks, logits, attn = self(x)
        #     loss = F.cross_entropy(logits, y.squeeze())
        #     probs = torch.softmax(logits, dim=1)
        #     preds = torch.argmax(probs, dim=1)
  
        #     dsc = dice(preds, y.squeeze(), average='macro', num_classes=self.num_classes, ignore_index=0)

        return loss, dsc, probs, preds, x, recons, masks, attn

    def training_step(self, batch, batch_idx):
        loss, dsc, probs, preds, imgs, recons, masks, attn = self.process_batch(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice", dsc, prog_bar=True)

        if self.train_imgs is None:
            self.train_imgs = imgs[:5].cpu()
            self.train_preds = preds[:5].cpu()
            self.train_recons = recons[:5].cpu()
            self.train_masks = masks[:5].cpu()
            self.train_attn = attn[:5].cpu()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, dsc, _, preds, imgs, recons, masks, attn = self.process_batch(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_dice", dsc, prog_bar=True, sync_dist=True)

        if self.val_imgs is None:
            # randomly select 5 images to log
            indices = torch.randint(0, imgs.shape[0], (5,))
            self.val_imgs = imgs[indices].cpu()
            self.val_preds = preds[indices].cpu()
            self.val_recons = recons[indices].cpu()
            self.val_masks = masks[indices].cpu()
            self.val_attn = attn[indices].cpu()

    def on_test_start(self):
        self.test_probmaps = []
        self.test_predmaps = []

    def test_step(self, batch, batch_idx):
        loss, dsc, probs, preds = self.process_batch(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_dice", dsc)
        self.test_probmaps.append(probs)
        self.test_predmaps.append(preds)

    def on_train_epoch_end(self):
        if self.log_images:
            if not self.logged_train_images_this_epoch and self.train_imgs is not None:
                self._log_train_images()
                self.logged_train_images_this_epoch = True
            
            self.train_imgs = None
            self.train_preds = None
            self.logged_train_images_this_epoch = False

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.optimizers().param_groups[0]['lr']
        if self.log_images:
            self.logger.experiment.log({"learning_rate": lr}, commit=False)

    def on_validation_epoch_end(self):
        if self.log_images:
            if not self.logged_val_images_this_epoch and self.val_imgs is not None:
                self._log_val_images()
                self.logged_val_images_this_epoch = True
            
            self.val_imgs = None
            self.val_preds = None
            self.logged_val_images_this_epoch = False

            self._log_key_images()

    def test_step(self, batch, batch_idx):
        pass

    def _log_val_images(self):
        imgs, preds, recons, masks, attn = self.val_imgs, self.val_preds, self.val_recons, self.val_masks, self.val_attn
        # print(masks.shape, recons.shape) # shape (5, num_slots, res, res) (5, res, res)
        grid = make_grid(imgs)
        grid_val = make_grid(masks[0].view(self.num_slots, 1, self.image_resolution, self.image_resolution), nrow=self.num_slots).float()
        grid_recons = make_grid(recons).float()
        self.logger.experiment.log({
            "Validation Images": [
                wandb.Image(grid, caption="Original Images"),
                wandb.Image(grid_val, caption="Masks"),
                wandb.Image(grid_recons, caption="Reconstructed Images")
            ],
            "trainer/global_step": self.global_step,
        })

    def _log_train_images(self):
        imgs, preds, recons, masks, attn = self.train_imgs, self.train_preds, self.train_recons, self.train_masks, self.train_attn
        grid = make_grid(imgs)
        grid_val = make_grid(masks[0].view(self.num_slots, 1, self.image_resolution, self.image_resolution), nrow=self.num_slots).float()
        grid_recons = make_grid(recons).float()
        self.logger.experiment.log({
            "Train Images": [
                wandb.Image(grid, caption="Original Images"),
                wandb.Image(grid_val, caption="Masks"),
                wandb.Image(grid_recons, caption="Reconstructed Images"),
            ],
            "trainer/global_step": self.global_step,
        })

    def _log_key_images(self):
        recons, attn, masks, preds = self.val_recons, self.val_attn, self.val_masks, self.val_preds
        recons_grid = make_grid(recons)
        #print(attn.shape) #attn shape (batch_size, num_slots, embedding_shape[0], embedding_shape[1])
        attn = attn.reshape(-1, self.num_slots, self.embedding_shape[0], self.embedding_shape[1])
        attn_neg = 1 - attn
        attn = F.interpolate(attn, size=(self.image_resolution, self.image_resolution), mode='nearest')
        attn_neg = F.interpolate(attn_neg, size=(self.image_resolution, self.image_resolution), mode='nearest')
        attn_maps = make_grid(attn.reshape(-1, 1, self.image_resolution, self.image_resolution), nrow=self.num_slots)
        attn_neg_maps = make_grid(attn_neg.reshape(-1, 1, self.image_resolution, self.image_resolution), nrow=self.num_slots)
        # masks = make_grid(masks.reshape(-1, 1, self.image_resolution, self.image_resolution), nrow=self.num_slots).float()
   
        self.logger.experiment.log({
            "Reconstructed Images": [
                wandb.Image(recons_grid, caption="Reconstructed Images"),
            ],
            "trainer/global_step": self.global_step,
        })
        self.logger.experiment.log({
            "Slot Attention Neg Maps": [
                wandb.Image(attn_neg_maps, caption="Attention Maps"),
            ],
            "trainer/global_step": self.global_step,
        })
        self.logger.experiment.log({
            "Slot Attention Maps": [
                wandb.Image(attn_maps, caption="Attention Maps"),
            ],
            "trainer/global_step": self.global_step,
        })
        # self.logger.experiment.log({
        #     "Masks": [
        #         wandb.Image(masks, caption="Masks"),

        #     ],
        # })
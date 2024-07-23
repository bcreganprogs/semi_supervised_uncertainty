import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from torchmetrics.functional import dice
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from pytorch_lightning import LightningModule
import wandb

from utils.utils import SoftPositionEmbed, spatial_broadcast, unstack_and_split
from models.slot_attention import ProbabalisticSlotAttention, FixedSlotAttention, SlotAttention, FixedSlotAttentionMultiHead
from models.decoders import SlotSpecificDecoder, Decoder

class ObjectSpecificSegmentation(LightningModule):
    def __init__(self, encoder, num_slots, num_iterations, num_classes, slot_dim=128, task='recon', include_seg_loss=False, slot_attn_type='standard', decoder_type='slot_specific', 
                learning_rate=1e-3, image_chans=1, embedding_dim=768, image_resolution=224, softmax_class=True, temperature=1, freeze_encoder=False, lr_warmup=True, log_images=True,
                embedding_shape=None, multi_truth=False):
        super(ObjectSpecificSegmentation, self).__init__()

        self.encoder = encoder
        self.embedding_shape = (image_resolution // 2**4, image_resolution // 2**4) if embedding_shape is None else embedding_shape#(image_resolution // 16, image_resolution // 16)
        self.encoder_pos_embeddings = SoftPositionEmbed(embedding_dim, self.embedding_shape)

        # slot attention
        if slot_attn_type == 'standard':
            self.slot_attention = SlotAttention(num_slots, slot_dim, num_iterations, temperature=temperature)
        elif slot_attn_type == 'probabilistic':
            self.slot_attention = ProbabalisticSlotAttention(num_slots=num_slots, dim=slot_dim, num_iterations=num_iterations, temperature=temperature)
        elif slot_attn_type == 'fixed':
            self.slot_attention = FixedSlotAttention(num_slots=num_slots, dim=slot_dim, num_iterations=num_iterations, temperature=temperature)
        else:
            self.slot_attention = FixedSlotAttentionMultiHead(num_slots, slot_dim, num_iterations, temperature=temperature, probabalistic=False)
            
        # decoder
        self.decoder_type = decoder_type
        if decoder_type == 'slot_specific':
            if task == 'recon':
                self.decoder = SlotSpecificDecoder(slot_dim, num_slots, num_classes, include_recon=True, softmax_class=softmax_class, image_chans=image_chans, decoder_type='slot_specific',
                                                   resolution=image_resolution)
            else:
                self.decoder = SlotSpecificDecoder(slot_dim, num_slots, num_classes, include_recon=False, softmax_class=softmax_class, image_chans=image_chans, decoder_type='slot_specific',
                                                   resolution=image_resolution)
        else:
            self.decoder = SlotSpecificDecoder(slot_dim, num_slots, num_classes, include_recon=True, softmax_class=softmax_class, image_chans=image_chans, decoder_type='shared',
                                               resolution=image_resolution)
        self.decoder_pos_embeddings = SoftPositionEmbed(slot_dim, (8, 8))

        # hyperparameters
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.image_chans = image_chans
        self.task = task
        self.learning_rate = learning_rate
        self.lr_warmup = lr_warmup
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.include_seg_loss = include_seg_loss
        self.image_resolution = image_resolution
        self.multi_truth = multi_truth

        # classification head
        self.softmax_class = softmax_class
  
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2, slot_dim),
        )

        self.decoder.apply(self.init_weights)
        self.mlp.apply(self.init_weights)
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
        try:
            patch_embeddings, _, _ = self.encoder.model.forward_encoder(x, 0.0)   # shape (batch_size, num_patches, embed_dim)
            patch_embeddings = patch_embeddings[:, 1:, :]       # exclude cls token
            
        except:
            patch_embeddings = self.encoder(x)    

        
        # add position embeddings
        # patch_embeddings = patch_embeddings.reshape(batch_size, self.embedding_shape[0], self.embedding_shape[1], -1)     # combine patches into image
        # patch_embeddings = self.encoder_pos_embeddings(patch_embeddings)   # shape (batch_size, 14, 14, embed_dim)
        # patch_embeddings = patch_embeddings.reshape(batch_size, int(self.embedding_shape[0] * self.embedding_shape[1]), -1)   # shape (batch_size, num_patches, embed_dim)
        
        patch_embeddings = self.embedding_norm(patch_embeddings)
        patch_embeddings = self.mlp(patch_embeddings)  # shape (batch_size*num_patches, embed_dim)

        slots, attn = self.slot_attention(patch_embeddings)  # shape (batch_size, num_slots, slot_dim)
        
        #x = spatial_broadcast(slots, self.decoder.decoder_initial_size)  # shape (batch_size*num_slots, width_init, height_init, slot_dim)
        # spatial broadcast
        x = slots.view(-1, slots.shape[-1])[:, :, None, None]
        x = x.repeat(1, 1, *self.decoder.decoder_initial_size)  # shape (batch_size*num_slots, slot_dim, width_init, height_init
        x = x.permute(0, 2, 3, 1)  # shape (batch_size*num_slots, width_init, height_init, slot_dim)
     
        if self.task == 'recon':
            x = self.decoder_pos_embeddings(x)
            x = x.view(batch_size, slots.shape[1], 8, 8, x.shape[-1])
            x = x.permute(0, 1, 4, 2, 3) # shape (batch_size, num_slots, slot_dim, width_init, height_init)
            x = self.decoder(x).to(patch_embeddings.device)     # shape (batch_size, num_classes, H, W)
            decoded, masks = torch.split(x, [self.image_chans, 1], dim=2)
            masks = F.softmax(masks, dim=1)         # softmax over self.num_slotslasses
            recons = torch.sum(decoded * masks, dim=1)
            
        else: # segmentation
            if self.decoder_type == 'slot_specific':
                x = self.decoder_pos_embeddings(x)     # shape (batch_size, num_classes, H, W)
                x = x.view(batch_size, slots.shape[1], 8, 8, x.shape[-1])
                x = x.permute(0, 1, 4, 2, 3)  # shape (batch_size, num_slots, slot_dim, width_init, height_init)
                x = self.decoder(x)  # shape (batch_size, num_classes, 224, 224)
                recons, masks = x, None
            else:
                x = self.decoder_pos_embeddings(x)     # shape (batch_size, num_classes, H, W)
                x = self.decoder(x).to(patch_embeddings.device)     # shape (batch_size, num_classes, H, W)
                decoded, masks = torch.split(x, [self.num_slots, 1], dim=2)
                recons = decoded
                masks = masks

        return recons, masks, attn
    
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Parameter)):
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
    
    def warmup_lr_scheduler(self, optimizer, warmup_steps, decay_steps, start_lr, target_lr):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps)) * (target_lr - start_lr) / target_lr + start_lr / target_lr
            else:
                return 0.5 ** ((current_step - warmup_steps) / decay_steps)
        return LambdaLR(optimizer, lr_lambda)

    def process_batch(self, batch, batch_idx):

        if self.task == 'recon':
            if self.include_seg_loss:
                x, y = batch['image'], batch['labelmap']
            else:
                x = batch['image']
            
            x = x.float() 
            recons, masks, attn = self(x)
            if self.softmax_class:
                mask_probs = torch.softmax(masks, dim=1)
                preds = torch.argmax(mask_probs, dim=1)
            else: # use mlp predictor
                mask_probs = masks
                preds = torch.round(mask_probs)

            recon_loss = F.mse_loss(recons.squeeze(), x.squeeze())      #recon loss
            if self.include_seg_loss:
                seg_loss = F.cross_entropy(masks.squeeze(), y.squeeze())     # segmentation loss
                loss = seg_loss
                dsc = dice(preds, y.squeeze(), average='macro', num_classes=self.num_classes, ignore_index=0)
            else:
                loss = recon_loss
                dsc = 0

            probs = torch.softmax(recons, dim=1)
            
        else:
            x, y = batch['image'], batch['labelmap']
            logits, masks, attn = self(x)
            loss = F.cross_entropy(logits, y.squeeze())
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
  
            dsc = dice(preds, y.squeeze(), average='macro', num_classes=self.num_classes, ignore_index=0)

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
            self.val_imgs = imgs[:5].cpu()
            self.val_preds = preds[:5].cpu()
            self.val_recons = recons[:5].cpu()
            self.val_masks = masks[:5].cpu()
            self.val_attn = attn[:5].cpu()

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
        grid = make_grid(imgs)
        grid_val = make_grid(masks[0]).float()
        #make_grid(preds).float()
        grid_recons = make_grid(recons).float()
        self.logger.experiment.log({
            "Validation Images": [
                wandb.Image(grid, caption="Original Images"),
                wandb.Image(grid_val, caption="Masks"),
                wandb.Image(grid_recons, caption="Reconstructed Images")
            ],
        })

    def _log_train_images(self):
        imgs, preds, recons, masks, attn = self.train_imgs, self.train_preds, self.train_recons, self.train_masks, self.train_attn
        grid = make_grid(imgs)
        grid_val = make_grid(masks[0]).float()
        #make_grid(preds).float()
        grid_recons = make_grid(recons).float()
        self.logger.experiment.log({
            "Train Images": [
                wandb.Image(grid, caption="Original Images"),
                wandb.Image(grid_val, caption="Masks"),
                wandb.Image(grid_recons, caption="Reconstructed Images")
            ],
        })

    def _log_key_images(self):
        recons, attn, masks, preds = self.val_recons, self.val_attn, self.val_masks, self.val_preds
        recons_grid = make_grid(recons)
        attn_maps = make_grid(attn.reshape(-1, 1, 14, 14), nrow=self.num_slots)
        masks = make_grid(masks.reshape(-1, 1, 14, 14), nrow=self.num_slots).float()
   
        self.logger.experiment.log({
            "Reconstructed Images": [
                wandb.Image(recons_grid, caption="Reconstructed Images"),
            ],
        })
        self.logger.experiment.log({
            "Slot Attention Maps": [
                wandb.Image(attn_maps, caption="Attention Maps"),
            ],
        })
        # self.logger.experiment.log({
        #     "Masks": [
        #         wandb.Image(masks, caption="Masks"),

        #     ],
        # })
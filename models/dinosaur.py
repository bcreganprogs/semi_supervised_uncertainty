import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torchmetrics.functional import dice
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from pytorch_lightning import LightningModule
import wandb

from utils.utils import SoftPositionEmbed, spatial_broadcast, unstack_and_split
from models.slot_attention import ProbabalisticSlotAttention, FixedSlotAttention, SlotAttention
from models.decoders import SlotSpecificDecoder, Decoder, MlpDecoder

class DINOSAUR(LightningModule):
    def __init__(self, frozen_encoder, trainable_encoder, num_slots, num_iterations, num_classes, slot_dim=128, task='recon', include_seg_loss=False, probabilistic_slots=True, 
                dec_type='mlp', learning_rate=1e-3, hidden_decoder_dim=2048, temperature=1, lr_warmup=True, log_images=True):
        super(DINOSAUR, self).__init__()

        self.frozen_encoder = frozen_encoder.model
        self.trainable_encoder = trainable_encoder.model
        self.dec_type = dec_type
        self.encoder_pos_embeddings = SoftPositionEmbed(slot_dim, (14, 14))
        if probabilistic_slots:
            self.slot_attention = ProbabalisticSlotAttention(num_slots=num_slots, dim=slot_dim, num_iterations=num_iterations, temperature=temperature)
        else:
            #self.slot_attention = SlotAttention(num_slots, slot_dim, num_iterations, temperature=temperature)
            self.slot_attention = FixedSlotAttentionMultiHeadProb(num_slots=num_slots, dim=slot_dim, num_iterations=num_iterations, temperature=temperature)
            #FixedSlotAttention(num_slots=num_slots, dim=slot_dim, num_iterations=num_iterations, temperature=temperature)
        
        if self.dec_type=='transformer':
            self.dec = TransformerDecoder(
                args.num_dec_blocks, args.max_tokens, args.d_model, args.num_heads, args.dropout, args.num_cross_heads)
            if self.cappa > 0:
                assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')   
                self.mask_token = nn.Parameter(torch.zeros(1, 1, args.d_model))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, args.d_model))
                torch.nn.init.normal_(self.pos_embed, std=.02)
                torch.nn.init.normal_(self.mask_token, std=.02)
        elif self.dec_type=='mlp':
            self.mlp_decoder = MlpDecoder(slot_dim, 768, 14*14, hidden_features=hidden_decoder_dim)

        self.mlp = nn.Sequential(
            nn.Linear(768, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        self.decoder_pos_embeddings = SoftPositionEmbed(slot_dim, (14, 14))
        self.num_classes = num_classes
        self.task = task
        self.learning_rate = learning_rate
        self.lr_warmup = lr_warmup
        self.embedding_norm = nn.LayerNorm(768)
        self.embedding_norm_decoder = nn.LayerNorm(slot_dim)

        self.include_seg_loss = include_seg_loss

        self.attn = None
        self.masks = None
        self.recons = None
        self.preds = None

        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
  
        self.mlp_decoder.apply(self.init_weights)
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
        self.val_imgs = None
        self.val_preds = None
        self.logged_train_images_this_epoch = False
        self.logged_val_images_this_epoch = False

    def forward(self, x):
        batch_size = x.size(0)
        with torch.no_grad():
            patch_embeddings_frozen, _, _ = self.frozen_encoder.forward_encoder(x, 0.0)
            patch_embeddings_frozen = patch_embeddings_frozen[:, 1:, :]     # exclude cls token

        # generate patch features from trainable encoder
        patch_embeddings_train, _, _ = self.trainable_encoder.forward_encoder(x, 0.0)
        patch_embeddings_train = patch_embeddings_train[:, 1:, :]           # exclude cls token

        # intermediate mlp between encoder and slot attention
        patch_embeddings_train = self.embedding_norm(patch_embeddings_train)
        patch_embeddings_train = self.mlp(patch_embeddings_train)                       # shape (batch_size*num_patches, embed_dim)

        # apply slot attention to trainable patch features
        slots, slot_attn = self.slot_attention(patch_embeddings_train)            # slots: shape (batch_size, num_slots, slot_dim)

        x = self.embedding_norm_decoder(slots)
        recons, masks = self.mlp_decoder(x)                                       # shape (batch_size, num_classes, H, W, slot_dim + 1)

        # log masks
        self.masks = masks[0, ...]

        preds = torch.argmax(masks, dim=1)

        return patch_embeddings_frozen, recons, masks, preds, slot_attn
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        if not self.lr_warmup:
            return optimizer

        scheduler = self.warmup_lr_scheduler(optimizer, 10000, 100000, 1e-6, self.learning_rate)
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
        #x, y = batch['image'], batch['labelmap']
        x = batch['image']
        try:
            y = batch['labelmap']
        except:
            y = None
     
        # generate patch features from frozen encoder
        targets, recons, masks, preds, attn = self(x)

        # log masks and attn
        self.attn = attn[0, ...]
        self.recons = recons[0, ...]
        self.preds = preds[0, ...]

        # calculate loss
        loss = torch.cdist(targets, recons, p=2).mean()
        if self.include_seg_loss and y is not None:
            seg_loss = F.cross_entropy(masks, y.squeeze(), ignore_index=0)
            loss += seg_loss
            dsc = dice(preds, y.squeeze(), average='macro', num_classes=self.num_classes, ignore_index=0)
        else:
            dsc = 1.0
        
        return loss, dsc, preds, x

    def training_step(self, batch, batch_idx):
        loss, dsc, preds, imgs = self.process_batch(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice", dsc, prog_bar=True)

        if self.train_imgs is None:
            self.train_imgs = imgs[:5].cpu()
            self.train_preds = preds[:5].cpu()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, dsc, preds, imgs = self.process_batch(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_dice", dsc, prog_bar=True, sync_dist=True)

        if self.val_imgs is None:
            self.val_imgs = imgs[:5].cpu()
            self.val_preds = preds[:5].cpu()

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
                self._log_train_images(self.train_imgs, self.train_preds)
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
                self._log_val_images(self.val_imgs, self.val_preds)
                self.logged_val_images_this_epoch = True
            
            self.val_imgs = None
            self.val_preds = None
            self.logged_val_images_this_epoch = False

            self._log_key_images()

    def test_step(self, batch, batch_idx):
        pass

    def _log_val_images(self, imgs: torch.Tensor, preds: torch.Tensor):
        grid = make_grid(imgs)
        #grid_val = make_grid(preds).unsqueeze(1).float()
        self.logger.experiment.log({
            "Validation Images": [
                wandb.Image(grid, caption="Original Images"),
                #wandb.Image(grid_val, caption="Features")
            ],
        })

    def _log_train_images(self, imgs: torch.Tensor, preds: torch.Tensor):
        grid = make_grid(imgs)
        #grid_val = make_grid(preds).unsqueeze(1).float()
        self.logger.experiment.log({
            "Train Images": [
                wandb.Image(grid, caption="Original Images"),
                #wandb.Image(grid_val, caption="Features")
            ],
        })

    def _log_key_images(self):
        attn, masks = self.attn, self.masks
        attn_maps = make_grid(attn.reshape(-1, 14, 14)).unsqueeze(1)
        masks = make_grid(masks.reshape(-1, 14, 14)).unsqueeze(1)
      
        self.logger.experiment.log({
            "Slot Attention Maps": [
                wandb.Image(attn_maps, caption="Attention Maps"),
            ],
        })
        self.logger.experiment.log({
            "Masks": [
                wandb.Image(masks, caption="Alpha Masks"),
            ],
        })
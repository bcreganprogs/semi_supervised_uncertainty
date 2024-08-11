from functools import partial
from pathlib import Path
# import hydra
# from omegaconf import DictConfig

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from models.mae.mae import ViTAE
from models.segmentation import ObjectSpecificSegmentation
from models.data import JSRTDataModule, CheXpertDataModule, CLEVRNDataset, LIDCDataModule, SynthCardDataModule, CURVASDataModule
from models.encoders import CNNEncoder, ResNet34_8x8, get_resnet34_encoder

seed_everything(42, workers=True)

# saved_model = ViTAE.load_from_checkpoint('/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/bash_scripts/lightning_logs/chestxray_mae/chestxray_mae/gn9nzdz8/checkpoints/epoch=503-step=95256.ckpt',
#     model_kwargs={
#         'img_size': 224,
#         'embed_dim': 768,
#         'in_chans': 1,
#         'num_heads': 12,
#         'depth': 12,
#         'decoder_embed_dim': 512,
#         'decoder_depth': 8,
#         'decoder_num_heads': 16,
#         'norm_layer': partial(nn.LayerNorm, eps=1e-6),
#         'mlp_ratio': 4.0,
#         'patch_size': 16,
#         'norm_pix_loss': False,
#     },
#     learning_rate=1e-4,
#     map_location=torch.device('cpu'),
#     )

vit_model = ViTAE(
    model_kwargs={
        'img_size': 128,
        'embed_dim': 512,
        'in_chans': 1,          # 3 for clevr
        'num_heads': 4,
        'depth': 6,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'mlp_ratio': 4.0,
        'patch_size': 8,
        'norm_pix_loss': False,
    },
    learning_rate=1e-4,
)

oss = ObjectSpecificSegmentation(vit_model, num_slots=4, num_iterations=3, num_classes=4, slot_dim=512, task='recon', decoder_type='shared', 
                                 learning_rate=4e-4, freeze_encoder=False, temperature=1, log_images=True, lr_warmup=True, include_seg_loss=False,
                                 softmax_class=False, slot_attn_type='probabilistic', image_chans=1, image_resolution=128, embedding_dim=512,
                                 embedding_shape=(16, 16), include_mlp=True, include_pos_embed=False)


#cnnencoder = CNNEncoder(slot_dim=128, num_channels=1)

# oss = ObjectSpecificSegmentation(cnnencoder, num_slots=7, num_iterations=3, num_classes=4, slot_dim=128, task='recon', decoder_type='shared', 
#                                  learning_rate=4e-4, freeze_encoder=False, temperature=1, log_images=True, lr_warmup=True, include_seg_loss=False,
#                                  softmax_class=True, slot_attn_type='multi-head', image_chans=3, embedding_dim=128)

# resnet34_8x8 = ResNet34_8x8(get_resnet34_encoder())         # output is shape (batch_size, 256, 8, 8)

# oss = ObjectSpecificSegmentation(resnet34_8x8, num_slots=4, num_iterations=3, num_classes=4, slot_dim=512, task='recon', decoder_type='shared', 
#                                  learning_rate=4e-4, freeze_encoder=False, temperature=1, log_images=True, lr_warmup=True, include_seg_loss=True,
#                                  softmax_class=False, slot_attn_type='probabilistic', image_chans=1, image_resolution=512, embedding_dim=512,
#                                  embedding_shape=(16, 16), include_mlp=True, include_pos_embed=True)

#data = JSRTDataModule(data_dir='/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/data/JSRT/', batch_size=64, augmentation=True)
#data = CheXpertDataModule(data_dir='/vol/biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224/', batch_size=64, cache=True)
#data = CLEVRNDataset(batch_size=128, input_res=224)
#data = LIDCDataModule(batch_size=64, cache=True)
data = SynthCardDataModule(batch_size=32, rate_maps=1.0)
#data = CURVASDataModule(batch_size=56, cache=True)

wandb_logger = WandbLogger(save_dir='./runs/lightning_logs/synthetic_cardiac/', project='synthetic_cardiac',
                           name='4head_vit_recon', id='4head_vit_recon', offline=False)
output_dir = Path(f"synthetic_cardiac/run_{wandb_logger.experiment.id}")  # type: ignore
print("Saving to" + str(output_dir.absolute()))

# wandb_logger = WandbLogger(save_dir='./runs/lightning_logs/abd_seg/', project='abd_seg',
#                            name='prob_4head_resnet_seg_4', id='prob_4head_resnet_seg_4', offline=False)
# output_dir = Path(f"abd_seg/run_{wandb_logger.experiment.id}")  # type: ignore
# print("Saving to" + str(output_dir.absolute()))

trainer = Trainer(
    max_epochs=5000,
    precision='16-mixed',
    accelerator='auto',
    devices=[0],
    strategy='ddp_find_unused_parameters_true',
    # accumulate batches
    #accumulate_grad_batches=2,
    # log_every_n_steps=250,
    #val_check_interval=0.5,
    check_val_every_n_epoch=10,
    logger=wandb_logger,
    callbacks=[ModelCheckpoint(monitor="val_loss", mode='min'), TQDMProgressBar(refresh_rate=20)],
)

torch.set_float32_matmul_precision('medium')

trainer.fit(model=oss, datamodule=data)


# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     # Create your CNN encoder here based on cfg if needed
#     encoder = get_encoder(cfg)
    
#     model = ObjectSpecificSegmentation(
#         encoder,
#         num_slots=cfg.model.num_slots,
#         num_iterations=cfg.model.num_iterations,
#         num_classes=cfg.model.num_classes,
#         slot_dim=cfg.model.slot_dim,
#         task=cfg.model.task,
#         decoder_type=cfg.model.decoder_type,
#         learning_rate=cfg.model.learning_rate,
#         freeze_encoder=cfg.model.freeze_encoder,
#         temperature=cfg.model.temperature,
#         log_images=cfg.model.log_images,
#         lr_warmup=cfg.model.lr_warmup,
#         include_seg_loss=cfg.model.include_seg_loss,
#         softmax_class=cfg.model.softmax_class,
#         slot_attn_type=cfg.model.slot_attn_type,
#         image_chans=cfg.model.image_chans,
#         image_resolution=cfg.model.image_resolution,
#         embedding_dim=cfg.model.embedding_dim
#     )

#     data = get_data_module(cfg)

#     trainer = Trainer(
#         max_epochs=1000,
#         precision='16-mixed',
#         accelerator='auto',
#         devices=[0],
#         strategy='ddp_find_unused_parameters_true',
#         # log_every_n_steps=250,
#         val_check_interval=0.5,
#         #check_val_every_n_epoch=2,
#         #logger=wandb_logger,
#         callbacks=[ModelCheckpoint(monitor="val_loss", mode='min'), TQDMProgressBar(refresh_rate=20)],
#     )

#     torch.set_float32_matmul_precision('medium')

#     trainer.fit(model=oss, datamodule=data)

# if __name__ == "__main__":
#     main()
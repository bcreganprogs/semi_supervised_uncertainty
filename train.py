import os
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from models.data import JSRTDataModule, CheXpertDataModule

from medmnist.info import INFO
from medmnist.dataset import MedMNIST
from torch.utils.data import DataLoader

from models.mae.mae import ViTAE

#seed_everything(42, workers=True)

# data = JSRTDataModule(data_dir='./data/JSRT/', batch_size=64)
data = CheXpertDataModule(data_dir='/vol/biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224/', batch_size=512, cache=True)

# ViT base
# model = ViTAE(
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
#         'mask_ratio': 0.75,
#     },
#     learning_rate=1e-4,
# )

# ViT large
# model = ViTAE(
#     model_kwargs={
#         'img_size': 224,
#         'embed_dim': 1024,
#         'num_channels': 1,
#         'num_heads': 16,
#         'depth': 24,
#         'decoder_embed_dim': 512,
#         'decoder_depth': 8,
#         'decoder_num_heads': 16,
#         'norm_layer': partial(nn.LayerNorm, eps=1e-6),
#         'mlp_ratio': 4.0,
#         'patch_size': 16,
#         'norm_pix_loss': False,
#         'mask_ratio': 0.75,
#         'dropout': 0.00,
#     },
#     learning_rate=1e-4,
# )

saved_model = ViTAE.load_from_checkpoint('/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/bash_scripts/lightning_logs/chestxray_mae/chestxray_mae/gn9nzdz8/checkpoints/epoch=503-step=95256.ckpt',
     model_kwargs={
        'img_size': 224,
        'embed_dim': 768,
        'in_chans': 1,
        'num_heads': 12,
        'depth': 12,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'mlp_ratio': 4.0,
        'patch_size': 16,
        'norm_pix_loss': False,
    },
    learning_rate=1e-4,
     map_location=torch.device('cpu'),
)

torch.set_float32_matmul_precision("medium")

wandb_logger = WandbLogger(save_dir='./lightning_logs/chestxray_mae/', project='chestxray_mae')
output_dir = Path(f"chestxray_mae/run_{wandb_logger.experiment.id}")  # type: ignore
print("Saving to" + str(output_dir.absolute()))

#wandb_logger.watch(model, log="all", log_freq=100)

# wandb_logger.log_hyperparams(
#     OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
# )

trainer = Trainer(
    max_epochs=1000,
    #accumulate_grad_batches=2,
    precision='16-mixed',
    accelerator='auto',
    devices=[0],
    strategy="ddp",
    #log_every_n_steps=250,
    #val_check_interval=0.5,
    check_val_every_n_epoch=2,
    #save_top_k=1,
    logger=wandb_logger,
    callbacks=[ModelCheckpoint(monitor="val_loss", mode='min'), TQDMProgressBar(refresh_rate=250)],
)

trainer.fit(model=saved_model, datamodule=data)
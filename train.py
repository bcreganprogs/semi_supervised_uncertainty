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

from data import JSRTDataModule, CheXpertDataModule

from medmnist.info import INFO
from medmnist.dataset import MedMNIST
from torch.utils.data import DataLoader

from models.mae import ViTAE

#seed_everything(42, workers=True)

# data = JSRTDataModule(data_dir='./data/JSRT/', batch_size=64)
data = CheXpertDataModule(data_dir='/vol/biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224/', batch_size=64, cache=False)

# model = ViTAE.load_from_checkpoint('/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/lightning_logs/autoencoder/ViTAE/version_38/checkpoints/epoch=999-step=3000.ckpt',
#     model_kwargs={
#         'img_size': 224,
#         'embed_dim': 768,
#         'num_channels': 1,
#         'num_heads': 12,
#         'depth': 14,
#         'decoder_embed_dim': 512,
#         'decoder_depth': 8,
#         'decoder_num_heads': 16,
#         'norm_layer': nn.LayerNorm,
#         'mlp_ratio': 4.0,
#         'patch_size': 16,
#         'norm_pix_loss': False,
#         'dropout': 0.0,
#     },
#     learning_rate=1e-5
#     )

model = ViTAE(
    model_kwargs={
        'img_size': 224,
        'embed_dim': 1024,
        'num_channels': 1,
        'num_heads': 16,
        'depth': 18,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'mlp_ratio': 4.0,
        'patch_size': 16,
        'norm_pix_loss': False,
        'mask_ratio': 0.75,
        'dropout': 0.00,
    },
    learning_rate=1e-4,
)

# saved_model = ViTAE.load_from_checkpoint('/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/lightning_logs/version_94/checkpoints/epoch=1499-step=4500.ckpt',
#     model_kwargs={
#         'img_size': 224,
#         'embed_dim': 256,
#         'num_channels': 1,
#         'num_heads': 16,
#         'depth': 16,
#         'decoder_embed_dim': 128,
#         'decoder_depth': 10,
#         'decoder_num_heads': 16,
#         'norm_layer': nn.LayerNorm,
#         'mlp_ratio': 4.0,
#         'patch_size': 16,
#         'norm_pix_loss': False,
#         'mask_ratio': 0.7,
#         'dropout': 0.0,
#     },
#     learning_rate=1e-4,
#     )

torch.set_float32_matmul_precision("medium")

wandb_logger = WandbLogger(save_dir='./lightning_logs/chestxray_mae/', project='chestxray_mae')
output_dir = Path(f"chestxray_mae/run_{wandb_logger.experiment.id}")  # type: ignore
print("Saving to" + str(output_dir.absolute()))

#wandb_logger.watch(model, log="all", log_freq=100)

# wandb_logger.log_hyperparams(
#     OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
# )

trainer = Trainer(
    max_epochs=250,
    precision='16-mixed',
    accelerator='auto',
    devices=[0, 1],
    strategy="ddp",
    log_every_n_steps=250,
    val_check_interval=0.5,
    #save_top_k=1,
    logger=wandb_logger,
    callbacks=[ModelCheckpoint(monitor="val_loss", mode='max'), TQDMProgressBar(refresh_rate=100)],
)

trainer.fit(model=model, datamodule=data)
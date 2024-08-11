from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from models.mae.mae import ViTAE
from models.dinosaur import DINOSAUR
from models.data import JSRTDataModule, CheXpertDataModule, CLEVRNDataset, SynthCardDataModule


seed_everything(42, workers=True)

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

# saved_model2 = ViTAE.load_from_checkpoint('/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/bash_scripts/lightning_logs/chestxray_mae/chestxray_mae/gn9nzdz8/checkpoints/epoch=503-step=95256.ckpt',
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
#         'mask_ratio': 0.00,

#     },
#     learning_rate=1e-4,
#     map_location=torch.device('cpu'),
#     )

model2 = ViTAE(
    model_kwargs={
        'img_size': 224,
        'embed_dim': 768,
        'in_chans': 1,          # for clevr
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
)

oss = DINOSAUR(saved_model, model2, num_slots=6, num_iterations=3, num_classes=4, slot_dim=256, task='recon',
                                 learning_rate=4e-4, temperature=1, log_images=True, lr_warmup=True,
                                 probabilistic_slots=False)

#data = JSRTDataModule(data_dir='/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/data/JSRT/', batch_size=64, augmentation=True)
data = CheXpertDataModule(data_dir='/vol/biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224/', batch_size=128, cache=True)
#data = CLEVRNDataset(batch_size=32)
#data = SynthCardDataModule(batch_size=128, rate_maps=0.2)

wandb_logger = WandbLogger(save_dir='./runs/lightning_logs/dinosaur_recons/', project='dinosaur_recons',
                           name='chexpert_recon_mlp_decoder', id='chexpert_recon_mlp_decoder_3')
output_dir = Path(f"dinosaur_recons/run_{wandb_logger.experiment.id}")  # type: ignore
print("Saving to" + str(output_dir.absolute()))

trainer = Trainer(
    max_steps=250000,
    precision='16-mixed',
    accelerator='auto',
    devices=[0],
    strategy='ddp_find_unused_parameters_true',
    # log_every_n_steps=250,
    val_check_interval=0.5,
    #check_val_every_n_epoch=50,
    logger=wandb_logger,
    callbacks=[ModelCheckpoint(monitor="val_loss", mode='min'), TQDMProgressBar(refresh_rate=100)],
)

torch.set_float32_matmul_precision('medium')

trainer.fit(model=oss, datamodule=data)
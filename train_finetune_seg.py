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
from models.data import JSRTDataModule, CheXpertDataModule, CLEVRNDataset, LIDCDataModule, SynthCardDataModule
from models.encoders import CNNEncoder

seed_everything(42, workers=True)

cnnencoder = CNNEncoder(slot_dim=256, num_channels=1)

# oss = ObjectSpecificSegmentation.load_from_checkpoint('/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/bash_scripts/runs/lightning_logs/synth_cardiac/synthetic_cardiac/9lt1xvea/checkpoints/epoch=1409-step=42300.ckpt',
#                                 encoder=cnnencoder, num_slots=4, num_iterations=3, num_classes=4, slot_dim=256, task='recon', decoder_type='shared', 
#                                 learning_rate=4e-4, freeze_encoder=False, temperature=1, log_images=True, lr_warmup=True, include_seg_loss=True,
#                                 softmax_class=False, slot_attn_type='multi-head', image_chans=1, image_resolution=128, embedding_dim=256,
#                                 embedding_shape=(16, 16), map_location='cpu')
oss = ObjectSpecificSegmentation.load_from_checkpoint('/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/bash_scripts/runs/lightning_logs/synth_cardiac/synthetic_cardiac/gak81oyh/checkpoints/epoch=1309-step=39300.ckpt',
                                encoder=cnnencoder, num_slots=4, num_iterations=3, num_classes=4, slot_dim=256, task='recon', decoder_type='shared', 
                                learning_rate=2e-4, freeze_encoder=False, temperature=1, log_images=True, lr_warmup=True, include_seg_loss=True,
                                softmax_class=False, slot_attn_type='probabilistic', image_chans=1, image_resolution=128, embedding_dim=256,
                                embedding_shape=(16, 16), map_location='cpu')

# oss.include_seg_loss = True
# # # # freeze oss decoder
oss.decoder.requires_grad_(True)
oss.encoder.requires_grad_(True)

#data = JSRTDataModule(data_dir='/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/data/JSRT/', batch_size=32, augmentation=True)
#data = CheXpertDataModule(data_dir='/vol/biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224/', batch_size=32, cache=True)
#data = CLEVRNDataset(batch_size=128, input_res=224)
#data = LIDCDataModule(batch_size=128, cache=True)
data = SynthCardDataModule(batch_size=128, rate_maps=1.0)

wandb_logger = WandbLogger(save_dir='./runs/lightning_logs/synth_cardiac/', project='synthetic_cardiac',
                           name='prob_40im_seg_gak81oyh_4', id='prob_40im_seg_gak81oyh_4')
# prob_40im_seg_gak81oyh
#4im_seg_9lt1xvea
output_dir = Path(f"synthetic_cardiac/run_{wandb_logger.experiment.id}")  # type: ignore
print("Saving to" + str(output_dir.absolute()))

trainer = Trainer(
    max_epochs=6000,
    precision='16-mixed',
    accelerator='auto',
    devices=[0],
    # accumulate batches
    #accumulate_grad_batches=2,
    # log_every_n_steps=250,
    limit_train_batches=2,
    #val_check_interval=0.5,
    check_val_every_n_epoch=100,
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
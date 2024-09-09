from functools import partial
from pathlib import Path
import argparse
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
from models.encoders import CNNEncoder, ResNet34_8x8, get_resnet34_encoder, DinoViT_16, Dinov2ViT_14, DinoViT_8, ResNet18, get_resnet18_encoder, DinoViTB_16

parser = argparse.ArgumentParser(description='Train models')

# data
parser.add_argument('--dataset', type=str, required=True, help='Dataset to run the model on')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--augmentation', action=argparse.BooleanOptionalAction, help='Augment data')
parser.add_argument('--cache', action=argparse.BooleanOptionalAction, help='Cache data')

# model
parser.add_argument('--load', type=str, default=None, help='path to model checkpoint')
parser.add_argument('--encoder', type=str, default='dinovit', help='Encoder type')
parser.add_argument('--num_slots', type=int, default=8, help='Number of slots')
parser.add_argument('--num_iterations', type=int, default=3, help='Number of iterations')
parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
parser.add_argument('--slot_dim', type=int, default=64, help='Slot dimension')
parser.add_argument('--decoder_type', type=str, default='shared', help='Decoder type')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--freeze_encoder', action=argparse.BooleanOptionalAction, help='Freeze encoder')
parser.add_argument('--temperature', type=float, default=1, help='Temperature')
parser.add_argument('--log_images', action=argparse.BooleanOptionalAction, help='Log images')
parser.add_argument('--lr_warmup', action=argparse.BooleanOptionalAction, help='Learning rate warmup')
parser.add_argument('--include_seg_loss', action=argparse.BooleanOptionalAction, help='Include segmentation loss')
parser.add_argument('--calibration_loss', action=argparse.BooleanOptionalAction, help='Include calibration loss')
parser.add_argument('--slot_attn_type', type=str, default='probabilistic', help='Slot attention type')
parser.add_argument('--probabilistic_sample', action=argparse.BooleanOptionalAction, help='Sample from probabilistic slot attention')
parser.add_argument('--image_chans', type=int, default=1, help='Number of image channels')
parser.add_argument('--image_resolution', type=int, default=512, help='Image resolution')
parser.add_argument('--embedding_dim', type=int, default=384, help='Embedding dimension')
parser.add_argument('--embedding_shape', type=int, default=32, help='Embedding shape')
parser.add_argument('--include_pos_embed', action=argparse.BooleanOptionalAction, help='Include positional embedding')
parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
parser.add_argument('--num_patches', type=int, default=1024, help='Number of patches')
parser.add_argument('--slot_attn_heads', type=int, default=4, help='Number of slot attention heads')
parser.add_argument('--decoder_blocks', type=int, default=6, help='Number of decoder blocks')
parser.add_argument('--decoder_heads', type=int, default=8, help='Number of decoder heads')
parser.add_argument('--decoder_dim', type=int, default=256, help='Decoder dimension')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing')
parser.add_argument('--autoregressive', action=argparse.BooleanOptionalAction, help='Use autoregressive decoding')

# logging
parser.add_argument('--wandb_save_dir', type=str, default='./runs/lightning_logs/abd_seg/', help='Wandb save directory')
parser.add_argument('--wandb_project', type=str, default='abd_seg', help='Wandb project name')
parser.add_argument('--wandb_name', type=str, default='cnn_add_seg_3', help='Wandb run name')
parser.add_argument('--wandb_id', type=str, default='cnn_add_seg_3', help='Wandb run ID')
parser.add_argument('--wandb_offline', action=argparse.BooleanOptionalAction, help='Run Wandb in offline mode')
parser.add_argument('--output_dir', type=str, default='abd_seg/run_', help='Output directory')

# trainer
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
parser.add_argument('--max_epochs', type=int, default=50000, help='Maximum number of epochs')
parser.add_argument('--precision', type=str, default='16-mixed', help='Precision for training')
parser.add_argument('--devices', type=int, default=1, help='Devices to use')
parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator to use')
parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_true', help='Training strategy')
parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Number of batches to accumulate gradients')
parser.add_argument('--log_every_n_steps', type=int, default=250, help='Log every n steps')
parser.add_argument('--check_val_every_n_epoch', type=int, default=10, help='Check validation every n epochs')
parser.add_argument('--monitor', type=str, default='val_loss', help='Metric to monitor for checkpointing')
parser.add_argument('--checkpoint_mode', type=str, default='min', help='Checkpoint mode')
parser.add_argument('--tqdm_refresh_rate', type=int, default=100, help='TQDM progress bar refresh rate')

args = parser.parse_args()

seed_everything(args.seed, workers=True)

if args.encoder == 'resnet34':
    encoder = ResNet34_8x8(get_resnet34_encoder())
elif args.encoder == 'resnet18':
    encoder = ResNet18(get_resnet18_encoder())
elif args.encoder == 'dino16':
    encoder = DinoViT_16()
elif args.encoder == 'dinob16':
    encoder = DinoViTB_16()
elif args.encoder == 'dino8':
    encoder = DinoViT_8()
elif args.encoder == 'mae':
    encoder = ViTAE.load_from_checkpoint('/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/bash_scripts/lightning_logs/chestxray_mae/chestxray_mae/gn9nzdz8/checkpoints/epoch=503-step=95256.ckpt',
    model_kwargs={
        'img_size': args.image_resolution,
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

if args.dataset == 'jsrt':
    data = JSRTDataModule(data_dir='/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/data/JSRT/', batch_size=args.batch_size, augmentation=args.augmentation,
                          random_seed=args.seed)
elif args.dataset == 'chexpert':
    data = CheXpertDataModule(data_dir='/vol/biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224/', batch_size=args.batch_size, cache=args.cache, augmentation=args.augmentation,
                              random_seed=args.seed)
elif args.dataset == 'synthcard':
    data = SynthCardDataModule(batch_size=args.batch_size, rate_maps=1.0, augmentation=args.augmentation, cache=args.cache,
                               random_seed=args.seed)
elif args.dataset == 'curvas':
    data = CURVASDataModule(batch_size=args.batch_size, cache=args.cache, augmentation=args.augmentation,
                            random_seed=args.seed)
elif args.dataset == 'clevr':
    data = CLEVRNDataset(batch_size=args.batch_size, input_res=224, random_seed=args.seed)
elif args.dataset == 'lidc':
    data = LIDCDataModule(batch_size=args.batch_size, cache=args.cache, random_seed=args.seed)

if args.load is None:
    model = ObjectSpecificSegmentation(encoder, num_slots=args.num_slots, num_iterations=args.num_iterations, num_classes=args.num_classes, 
                                    slot_dim=args.slot_dim, decoder_type=args.decoder_type, 
                                    learning_rate=args.learning_rate, freeze_encoder=args.freeze_encoder, temperature=args.temperature, 
                                    log_images=args.log_images, lr_warmup=args.lr_warmup, include_seg_loss=args.include_seg_loss,
                                    slot_attn_type=args.slot_attn_type, probabilistic_sample=args.probabilistic_sample, image_chans=args.image_chans, 
                                    image_resolution=args.image_resolution, 
                                    embedding_dim=args.embedding_dim, embedding_shape=(args.embedding_shape, args.embedding_shape),
                                    include_pos_embed=args.include_pos_embed, patch_size=args.patch_size, num_patches=args.num_patches,
                                    slot_attn_heads=args.slot_attn_heads, decoder_blocks=args.decoder_blocks, decoder_heads=args.decoder_heads, 
                                    decoder_dim=args.decoder_dim, autoregressive=args.autoregressive, label_smoothing=args.label_smoothing,
                                    calibration_loss=args.calibration_loss)
else:
    model = ObjectSpecificSegmentation.load_from_checkpoint(args.load,
                                    encoder=encoder)


wandb_logger = WandbLogger(
    save_dir='./runs/lightning_logs/' + args.wandb_project,
    project=args.wandb_project,
    name=args.wandb_name,
    id=args.wandb_id,
    offline=args.wandb_offline
)

devices = [i for i in range(args.devices)]

trainer = Trainer(
    max_epochs=args.max_epochs,
    precision=args.precision,
    accelerator=args.accelerator,
    devices=devices,
    strategy=args.strategy,
    accumulate_grad_batches=args.accumulate_grad_batches,
    log_every_n_steps=args.log_every_n_steps,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    logger=wandb_logger,
    callbacks=[
        ModelCheckpoint(monitor=args.monitor, mode=args.checkpoint_mode),
        TQDMProgressBar(refresh_rate=args.tqdm_refresh_rate)
    ]
)

torch.set_float32_matmul_precision('medium')

trainer.fit(model=model, datamodule=data)
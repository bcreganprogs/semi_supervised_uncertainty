from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from vanilla_unet import ImageSegmenter

from data import JSRTDataModule

seed_everything(42, workers=True)

data = JSRTDataModule(data_dir='./data/JSRT/', batch_size=5)

model = ImageSegmenter(output_dim=4, learning_rate=0.001)

trainer = Trainer(
    max_epochs=50,
    accelerator='auto',
    devices=1,
    log_every_n_steps=5,
    logger=TensorBoardLogger(save_dir='./lightning_logs/segmentation/', name='jsrt-single-conv'),
    callbacks=[ModelCheckpoint(monitor="val_loss", mode='min'), TQDMProgressBar(refresh_rate=10)],
)
trainer.fit(model=model, datamodule=data)

trainer.validate(model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)

trainer.test(model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)
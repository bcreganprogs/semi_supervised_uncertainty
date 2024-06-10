import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch_lightning import LightningModule
from torchmetrics.functional import dice#, iou, hausdorff_distance

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)            
        )

    def forward(self, x):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, y):

        # x is from last layer y is skip connection
        x = self.upsample(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]

        x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))
        x = torch.cat([y, x], dim=1)
        x = self.conv(x)

        return x
    
class EndConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EndConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )


    def forward(self, x):
        # use softplus to avoid negative values
        # this resents evidence over classes
        return F.softplus(self.conv(x))

class ENN_UNet(LightningModule):
    def __init__(self, output_dim: int, learning_rate: float = 0.001):
        super().__init__()
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.test_probmaps = []
        self.test_predmaps = []

        self.conv_block_start = ConvBlock(1, 32)
        self.down_block_1 = Downsample(32, 64)
        self.down_block_2 = Downsample(64, 128)
        self.down_block_3 = Downsample(128, 256)
        self.up_block_1 = Upsample(256, 128)
        self.up_block_2 = Upsample(128, 64)
        self.up_block_3 = Upsample(64, 16)
        self.conv_block_end = EndConvBlock(16, self.output_dim)


    def forward(self, x):

        x1 = self.conv_block_start(x)
        x2 = self.down_block_1(x1)
        x3 = self.down_block_2(x2)
        x4 = self.down_block_3(x3)
        x5 = self.up_block_1(x4, x3)
        x6 = self.up_block_2(x5, x2)
        x7 = self.up_block_3(x6, x1)
        evidences = self.conv_block_end(x7)

        print(evidences.shape)
        # increase each evidence tensor by 1 to avoid negative values
        alphas = evidences + torch.ones_like(evidences)
        strength = torch.sum(alphas, dim=1, keepdim=True)
        probabilities = alphas / strength

        return probabilities

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def process_batch(self, batch):
        x, y = batch['image'], batch['labelmap']
        logits = self(x)
        loss = F.cross_entropy(logits, y.squeeze())

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        dsc = dice(preds, y.squeeze(), average='macro', num_classes=self.output_dim, ignore_index=0)

        return loss, dsc, probs, preds

    def training_step(self, batch, batch_idx):
        loss, dsc, probs, preds = self.process_batch(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice", dsc, prog_bar=True)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
            self.logger.experiment.add_image('train_images', grid, self.global_step)

            grid = torchvision.utils.make_grid(batch['labelmap'][0:4, ...].type('torch.FloatTensor'), nrow=2, normalize=True)
            self.logger.experiment.add_image('train_labelmaps', grid, self.global_step)

            grid = torchvision.utils.make_grid(probs[0:4, 1:4, ...], nrow=2, normalize=True)
            self.logger.experiment.add_image('train_probmaps', grid, self.global_step)

            grid = torchvision.utils.make_grid(preds[0:4, ...].unsqueeze(1).type('torch.FloatTensor'), nrow=2, normalize=True)
            self.logger.experiment.add_image('train_predmaps', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dsc, _, _ = self.process_batch(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dsc, prog_bar=True)

    def on_test_start(self):
        self.test_probmaps = []
        self.test_predmaps = []

    def test_step(self, batch, batch_idx):
        loss, dsc, probs, preds = self.process_batch(batch)
        self.log("test_loss", loss)
        self.log("test_dice", dsc)
        self.test_probmaps.append(probs)
        self.test_predmaps.append(preds)
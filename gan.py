#    ___                     _  ___   _   _  _
#   |   \ ___ _ __ _ __  ___| |/ __| /_\ | \| |__ _ ___ _ _
#   | |) / _ \ '_ \ '_ \/ -_) | (_ |/ _ \| .` / _` / -_) '_|
#   |___/\___/ .__/ .__/\___|_|\___/_/ \_\_|\_\__, \___|_|
#            |_|  |_|                         |___/

"""
GENERATIVE ADVERSARIAL NETWORK
-- Coded by Wouter Durnez
"""

import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid
from torchvision.transforms import Resize, ToTensor, ToPILImage, Normalize

from helper import log, hi


class DoppelDataset(Dataset):
    """
    Dataset class for face data
    """

    def __init__(self, face_dir: str, transform=None):

        self.face_dir = face_dir
        self.face_paths = os.listdir(face_dir)
        self.transform = transform

    def __len__(self):

        return len(self.face_paths)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        face_path = os.path.join(self.face_dir, self.face_paths[idx])
        face = io.imread(face_path)

        sample = {'image': face}

        if self.transform:
            sample = self.transform(sample['image'])

        return sample


class DoppelDataModule(pl.LightningDataModule):

    def __init__(self, data_dir='../data/faces', batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = transforms.Compose([
            ToTensor(),
            Resize(100),
            Normalize(mean=(123.26290927634774, 95.90498110733365, 86.03763122875182),
                      std=(63.20679012922922, 54.86211954409834, 52.31266645797249))
        ])

    def setup(self, stage=None):
        # Initialize dataset
        doppel_data = DoppelDataset(face_dir=self.data_dir, transform=self.transforms)

        # Train/val/test split
        n = len(doppel_data)
        train_size = int(.8 * n)
        val_size = int(.1 * n)
        test_size = n - (train_size + val_size)

        self.train_data, self.val_data, self.test_data = random_split(dataset=doppel_data,
                                                                      lengths=[train_size, val_size, test_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


class DoppelGenerator(nn.Sequential):
    """
    Generator network that produces images based on latent vector
    """

    def __init__(self, latent_dim: int):
        super().__init__()

        def block(in_channels: int, out_channels: int, padding: int = 1, stride: int = 2, bias=False):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride,
                                   padding=padding, bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(True)
            )

        self.model = nn.Sequential(
            block(latent_dim, 512, padding=0, stride=1),
            block(512, 256),
            block(256, 128),
            block(128, 64),
            block(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


class DoppelDiscriminator(nn.Sequential):
    """
    Discriminator network that classifies images in two categories
    """

    def __init__(self):
        super().__init__()

        def block(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class DoppelGAN(pl.LightningModule):

    def __init__(self,
                 channels: int,
                 width: int,
                 height: int,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64,
                 **kwargs):

        super().__init__()
        self.save_hyperparameters()

        # Initialize networks
        # data_shape = (channels, width, height)
        self.generator = DoppelGenerator(latent_dim=self.hparams.latent_dim, )
        self.discriminator = DoppelDiscriminator()

        self.validation_z = torch.randn(8, self.hparams.latent_dim,1,1)

    def forward(self, input):
        return self.generator(input)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images = batch

        # Sample noise (batchsize, latent_dim,1,1)
        z = torch.randn(images.shape[0], self.hparams.latent_dim,1,1)

        # Train generator #todo: understand this - does a training step loop over batches and optimizers?
        if optimizer_idx == 0:

            # Generate images (call generator -- see forward -- on latent vector)
            self.generated_images = self(z)

            # Log sampled images (visualize what the generator comes up with)
            sample_images = self.generated_images[:6]
            grid = make_grid(sample_images)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # Ground truth result (ie: all fake)
            valid = torch.ones(images.size(0), 1)

            # Adversarial loss is binary cross-entropy
            generator_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'gen_loss': generator_loss}

            output = {
                'loss': generator_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output

        # Train discriminator: classify real from generated samples
        if optimizer_idx == 1:

            # How well can it label as real?
            valid = torch.ones(images.size(0), 1)
            real_loss = self.adversarial_loss(self.discriminator(images), valid)

            # How well can it label as fake?
            fake = torch.zeros(images.size(0), 1)
            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            # Discriminator loss is the average of these
            discriminator_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': discriminator_loss}
            output = {
                'loss': discriminator_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # Optimizers
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        # Return optimizers/schedulers (currently no scheduler)
        return [opt_g, opt_d], []

    def on_epoch_end(self):

        # Log sampled images
        sample_images = self(self.validation_z)
        grid = make_grid(sample_images)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


if __name__ == '__main__':
    hi()

    # Global parameter
    image_dim = 100
    latent_dim = 100
    batch_size = 128

    # Initialize dataset
    tfs = transforms.Compose([
        ToPILImage(),
        Resize(image_dim),
        ToTensor()
    ])
    face_dataset = DoppelDataset(face_dir='../data/faces', transform=tfs)

    # Initialize data module
    face_data_module = DoppelDataModule(batch_size=batch_size)

    # Build models
    generator = DoppelGenerator(latent_dim=latent_dim)
    discriminator = DoppelDiscriminator()
    doppelgan = DoppelGAN(batch_size=batch_size, channels=3, width=image_dim, height=image_dim, latent_dim=latent_dim)

    # Test generator
    x = torch.rand(batch_size, latent_dim, 1, 1)
    y = generator(x)
    log(f'Generator: x {x.size()} --> y {y.size()}')

    # Test discriminator
    x = torch.rand(batch_size, 3, 128, 128)
    y = discriminator(x)
    log(f'Discriminator: x {x.size()} --> y {y.size()}')

    # Fit GAN
    trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)
    trainer.fit(model=doppelgan, datamodule=face_data_module)


    '''image = y[0].detach().numpy()
    image = np.transpose(image,axes=[1,2,0])
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_axis_off()
    plt.show()'''

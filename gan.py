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
from torch.nn import Upsample
from pl_bolts.datamodules import CIFAR10DataModule
from skimage import io
from tensorboard import program
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, ToTensor, Normalize
from torchvision.utils import make_grid

from helper import log, hi, compute_gradient_penalty


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

    def __init__(self, data_dir='../data/faces', batch_size: int = 64, num_workers: int = 0,
                 image_shape: tuple = (100, 100)):
        super().__init__()

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = transforms.Compose([
            ToTensor(),
            #Normalize(
            #    mean=[.5, .5, .5],
            #    std=[.5, .5, .5],
            #),
        ])

    def setup(self, stage=None):
        # Initialize dataset
        self.doppel_data = DoppelDataset(face_dir=self.data_dir, transform=self.transforms)

        # Train/val/test split
        n = len(self.doppel_data)
        train_size = int(.8 * n)
        val_size = int(.1 * n)
        test_size = n - (train_size + val_size)

        self.train_data, self.val_data, self.test_data = random_split(dataset=self.doppel_data,
                                                                      lengths=[train_size, val_size, test_size])

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

        return dataloader


class DoppelGenerator(nn.Sequential):
    """
    Generator network that produces images based on latent vector
    """

    def __init__(self, latent_dim: int, **kwargs):
        super().__init__()

        def block(in_channels: int, out_channels: int, padding: int = 1, stride: int = 2, bias=False):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=padding,
                    bias=bias
                ),
                nn.BatchNorm2d(num_features=out_channels),
                # nn.LeakyReLU(0.2, inplace=True)
                nn.PReLU(out_channels)
            ).cuda()

        self.model = nn.Sequential(
            block(latent_dim, 512, padding=0, stride=1),
            block(512, 256),
            block(256, 128),
            block(128, 64),
            block(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.model.cuda()

    def forward(self, input):
        return self.model(input)


class DoppelDiscriminator(nn.Sequential):
    """
    Discriminator network that classifies images in two categories
    """

    def __init__(self, **kwargs):
        super().__init__()

        def block(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=False),
            ).cuda()

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
        self.model.cuda()

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

        self.opt_g = None
        self.opt_d = None

        self.step = 0

        # Save all keyword arguments as hyperparameters, accessible through self.hparams.X)
        self.save_hyperparameters()
        self.to(device)
        print('GAN on {0}'.format(self.device))

        # Initialize networks
        self.generator = DoppelGenerator(latent_dim=self.hparams.latent_dim).cuda()
        self.discriminator = DoppelDiscriminator().cuda()

        self.validation_z = torch.randn(48, self.hparams.latent_dim, 1, 1, device=device)

    def forward(self, input):
        return self.generator(input)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if cifar:
            images, _ = batch
        else:
            images = batch

        # Sample noise (batch_size, latent_dim,1,1)
        z = torch.randn(images.size(0), self.hparams.latent_dim, 1, 1, device=device)
        z = z.type_as(images)

        # train discriminator
        if optimizer_idx == 1:
            fake_images = self(z)

            # real images
            valid = torch.ones(images.size(0), 1, device=device)
            real_valid = self.adversarial_loss(self.discriminator(images), valid)
            # real_valid = -torch.mean(self.discriminator(images))

            # fake images
            fake = torch.zeros(images.size(0), 1, device=device)
            fake_valid = self.adversarial_loss(self.discriminator(self(z).detach()), fake)
            # fake_valid = torch.mean(self.discriminator(self(z).detach()))

            d_loss = (real_valid + fake_valid) / 2

            # Add Wasserstein penalty
            if 'wasserstein_gp' in self.hparams:
                # Compute Wasserstein penalty
                penalty = compute_gradient_penalty(discriminator, images.data, fake_images.data, device=device)
                d_loss += self.hparams.wasserstein_gp * penalty

            tqdm_dict = {'d_loss': d_loss}

            output = {
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output

        # Train generator
        if optimizer_idx == 0:
            self.step += 1
            self.opt_g.zero_grad()  # pl?
            # Train the generator every n_critic steps
            if self.step % self.hparams.ncritics == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                crit_fake_images = self(z)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images

                # Ground truth result (ie: all fake)
                valid = torch.ones(images.size(0), 1, device=device)
                g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
                # g_loss = -torch.mean(self.discriminator(self(z)))

                # Log sampled images (visualize what the generator comes up with)
                sample_images = [crit_fake_images[0],
                                 crit_fake_images[int(len(crit_fake_images) / 2)],
                                 crit_fake_images[len(crit_fake_images) - 1]]

                grid = make_grid(sample_images)
                self.logger.experiment.add_image('step_images', grid, self.step)

                tqdm_dict = {'g_loss': g_loss}

                output = {
                    'loss': g_loss,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                }
                return output


    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # Optimizers
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        # Return optimizers/schedulers (currently no scheduler)
        return [self.opt_g, self.opt_d], []

    def on_epoch_end(self):
        # Log sampled images
        sample_images = self(self.validation_z)
        self.logger.experiment.add_images('epoch_images', sample_images, self.current_epoch)


if __name__ == '__main__':
    hi()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Available device: {0}'.format(device.type))

    # Start Tensorboard on localhost:8888
    tb = program.TensorBoard()
    tb.configure(
        argv=[
            None,
            '--logdir', 'c:/Users/aduen/PycharmProjects/doppelganger/lightning_logs/',
            '--port', '8888',
            '--samples_per_plugin', 'images=10'
        ])
    url = tb.launch()

    # Global parameter
    image_dim = 128
    latent_dim = 128
    batch_size = 32
    lambda_gp = 20  # loss weight for the gradient penalty
    critic_steps = 5  # critical steps when generator is trained

    # Cifar?
    cifar = False

    # Initialize data module
    if cifar:
        # Initialize dataset
        tfs = transforms.Compose([
            # ToPILImage(),
            Resize(image_dim),
            ToTensor()
        ])
        doppel_data_module = CIFAR10DataModule('../data/cifar', train_transforms=tfs, num_workers=4)
    else:
        doppel_data_module = DoppelDataModule(
            data_dir='../data/cats/square',
            batch_size=batch_size,
            image_shape=(128, 128),
        )

    # Build models

    generator = DoppelGenerator(latent_dim=latent_dim).cuda()
    discriminator = DoppelDiscriminator().cuda()

    # Test generator
    x = torch.rand(batch_size, latent_dim, 1, 1, device=device)
    y = generator(x)
    log(f'Generator: x {x.size()} --> y {y.size()}')

    # Test discriminator
    x = torch.rand(batch_size, 3, 128, 128, device=device)
    y = discriminator(x)
    log(f'Discriminator: x {x.size()} --> y {y.size()}')

    # Build GAN
    doppelgan = DoppelGAN(
        batch_size=batch_size,
        channels=3,
        width=image_dim,
        height=image_dim,
        latent_dim=latent_dim,
        wasserstein_gp=lambda_gp,
        ncritics=critic_steps
    ).cuda()

    # Fit GAN
    trainer = pl.Trainer(gpus=1, max_epochs=4000, progress_bar_refresh_rate=1)
    trainer.fit(model=doppelgan, datamodule=doppel_data_module)

    '''
    image = y[0].detach().numpy()
    image = np.transpose(image, axes=[1, 2, 0])
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_axis_off()
    plt.show()
    '''

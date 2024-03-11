# Script to train GAN on the MNIST

# Torch Import
import torch
import torch.nn as nn
import torch.optim as optim

# Torch Vision Import
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Import Data with Torch & using Tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Importing our classes from GAN.py & config.py
from GAN import Generator, Discriminator, initialize_weights
from GAN_config import Configuration

####
# Import hyper parameters
CFG = Configuration()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LEARNING_RATE = CFG.learning_rate
BETAS = CFG.betas
BATCH_SIZE = CFG.batch_size
IMG_SIZE = CFG.img_size
CHANNELS_IMG = CFG.channels_img
Z_DIM = CFG.z_dim
EPOCHS = CFG.num_epochs
FEATURES_DISC = CFG.features_disc
FEATURES_GEN = CFG.features_gen

# Create our transform
transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5 for _ in range(CHANNELS_IMG)],
                                    [0.5 for _ in range(CHANNELS_IMG)]
                                )
                                ])

# Get data & create dataset
dataset = datasets.MNIST(root= 'GAN/dataset', train= True, transform= transform, download= True)

loader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle= True)

# Init our Dicriminator & Generator with weights
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
discriminator = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(generator)
initialize_weights(discriminator)

# Optimizers
opt_generator = optim.Adam(generator.parameters(), lr= LEARNING_RATE, betas= BETAS)
opt_discriminator = optim.Adam(discriminator.parameters(), lr= LEARNING_RATE, betas= BETAS)
criterion = nn.BCELoss()

# Adding Noise
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# Logger
writer_fake = SummaryWriter(f"GAN/logs/fake")
writer_real = SummaryWriter(f"GAN/logs/real")
steps = 0

# Train Zone
generator.train()
discriminator.train()

for epochs in range(EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = generator(noise)

        # Train discriminator : max log(Disc(x)) + log(Disc(Gen(z)))
        discriminator_real = discriminator(real).reshape(-1)
        loss_discriminator_real = criterion(discriminator_real, torch.ones_like(discriminator_real))

        discriminator_fake = discriminator(fake).reshape(-1)
        loss_discriminator_fake = criterion(discriminator_fake, torch.ones_like(discriminator_fake))

        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2

        discriminator.zero_grad()
        loss_discriminator.backward(retain_graph= True)
        opt_discriminator.step()

        # Train Generator min log(1 - Disc(Gen(z))) <-> max log(Disc(Gen(z)))
        output = discriminator(fake).reshape(-1)
        loss_generator = criterion(output, torch.ones_like(output))

        generator.zero_grad()
        loss_generator.backward()
        opt_generator.step()


        # Monitoring
        if batch_idx % 100 == 0:
            print(f"Epoch[{epochs} / {EPOCHS}] -> Btach {batch_idx} / {len(loader)}")
            print(f"Loss Discriminator = {loss_discriminator: .4f}")
            print(f"Loss Generator = {loss_generator: .4f}")
            print("=" * 20)

            with torch.no_grad():
                fake = generator(fixed_noise)
                # Take a sample real & fake
                img_grid_real = torchvision.utils.make_grid(real[: 32], normalize= True)
                img_grid_fake = torchvision.utils.make_grid(fake[: 32], normalize= True)

                writer_real.add_image("Real", img_grid_real, global_step= steps)
                writer_fake.add_image("Fake", img_grid_fake, global_step= steps)

            steps += 1

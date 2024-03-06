import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

## Set Device

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = datasets.CIFAR10(
    root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True)

## Hyperparameters

latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10

## Define Generator

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Adjusted the number of channels
            nn.BatchNorm2d(32, momentum=0.8),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

## Define Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 512),  # Increase the number of features here
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)

## Define & Initialize Generator and Discriminator

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

## Loss and Optimizer

adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(),
                         lr=lr, betas=(beta1, beta2))

## Training Loop

for epoch in range(num_epochs):
    for i, batch in enumerate(data_loader):
        # Convert List to Tensor
        real_imgs = batch[0].to(device)
        
        print("Shape of real_imgs:", real_imgs.shape)

        # Adversial Ground Truth
        valid = torch.ones(real_imgs.size(0), 1, device=device)
        fake = torch.zeros(real_imgs.size(0), 1, device=device)

        # Configure Input
        real_imgs = real_imgs.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample Noise as Generator Input
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)

        # Generate a Batch of Images
        fake_imgs = generator(z)

        # Make Discriminator's Prediction on Real and Fake Images
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Backward Pass and Optimize

        d_loss.backward()
        optimizer_D.step()
        
        # Print the shape of validity after the Discriminator forward pass
        validity = discriminator(real_imgs)
        print("Shape of validity:", validity.shape)

        # ---------------------
        # Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generate a Batch of Images
        gen_imgs = generator(z)
        print("Shape of gen_imgs:", gen_imgs.shape)

        # Adversarial Loss
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # Backward Pass and Optimize
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Progress Monitor
        # ---------------------
        if (i+1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(data_loader)}], discriminator_loss: {d_loss.item():.4f}, generator_loss: {g_loss.item():.4f}')

    # Save Generated Images
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(
                generated, nrow=4, normalize=True)
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.show()

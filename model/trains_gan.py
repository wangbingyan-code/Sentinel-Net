import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from torchvision.utils import save_image
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Build generator network
        self.model = nn.Sequential(
            # Input is a vector of length latent_dim
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 448 * 320 * 3),  # Output is a 448x320 3-channel image
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 448, 320)  # Shape: (batch_size, 3, 448, 320)
        return img


# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Build discriminator network
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input: (3, 448, 320)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),  # Output a scalar
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)  # Output shape: [batch_size, 1]

# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, normal_dir, anomaly_dir, transform=None):
        self.normal_images = self.load_images(normal_dir)
        self.anomaly_images = self.load_images(anomaly_dir)
        self.transform = transform
        
    def load_images(self, dir_path):
        """Load image path list"""
        image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(('png', 'jpg'))]
        return image_paths
    
    def __len__(self):
        return len(self.normal_images) + len(self.anomaly_images)
    
    def __getitem__(self, idx):
        if idx < len(self.normal_images):
            image_path = self.normal_images[idx]
            label = 1  # Label normal sample
        else:
            image_path = self.anomaly_images[idx - len(self.normal_images)]
            label = 0  # Label anomaly sample
        
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((448, 320)),  # Resize image
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Create dataset and dataloader
normal_dir = "../dataset/train/normal"
anomaly_dir = "../dataset/train/anomaly"
dataset = CustomImageDataset(normal_dir, anomaly_dir, transform=transform)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Hyperparameters
latent_dim = 100
batch_size = 32
epochs = 100
lr = 0.0002
beta1 = 0.5

# Initialize models
generator = Generator(latent_dim).cuda()
discriminator = Discriminator().cuda()

# Initialize optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Train GAN
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        
        # Move data to GPU
        real_imgs = imgs.cuda()
        labels = labels.cuda()

        # Labels
        valid = torch.ones(batch_size, 1).cuda()
        fake = torch.zeros(batch_size, 1).cuda()

        # -----------------
        #  Train Discriminator
        # -----------------
        optimizer_d.zero_grad()

        # Real sample loss
        real_pred = discriminator(real_imgs)
        d_loss_real = adversarial_loss(real_pred, valid)

        # Generate fake samples
        z = torch.randn(batch_size, latent_dim).cuda()
        fake_imgs = generator(z)
        
        # Fake sample loss
        fake_pred = discriminator(fake_imgs.detach())
        d_loss_fake = adversarial_loss(fake_pred, fake)

        # Backpropagation and optimization
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_g.zero_grad()

        # Generator loss
        gen_pred = discriminator(fake_imgs)
        g_loss = adversarial_loss(gen_pred, valid)

        # Backpropagation and optimization
        g_loss.backward()
        optimizer_g.step()

    # Print loss
    print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save generated images every few epochs
    if epoch % 10 == 0:
        z = torch.randn(64, latent_dim).cuda()
        gen_imgs = generator(z)
        gen_imgs = gen_imgs.detach().cpu()
        # save_image(gen_imgs, f"../output/{epoch}.png", nrow=8, normalize=True)
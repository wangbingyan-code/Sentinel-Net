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
        
        # 构建生成器网络
        self.model = nn.Sequential(
            # 输入是 latent_dim 长度的向量
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 448 * 320 * 3),  # 输出为 448x320 的三通道图像
            nn.Tanh()  # 图像输出范围为 [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 448, 320)  # 转换为 (batch_size, 3, 448, 320) 形状
        return img


# 定义判别器（Discriminator）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # 构建判别器网络
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 输入为 (3, 448, 320)
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
            
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),  # 输出一个标量
            nn.Sigmoid()  # 结果在 [0, 1] 范围内
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)  # 返回一个标量 [batch_size, 1]
# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, normal_dir, anomaly_dir, transform=None):
        self.normal_images = self.load_images(normal_dir)
        self.anomaly_images = self.load_images(anomaly_dir)
        self.transform = transform
        
    def load_images(self, dir_path):
        """加载图像路径列表"""
        image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(('png', 'jpg'))]
        return image_paths
    
    def __len__(self):
        return len(self.normal_images) + len(self.anomaly_images)
    
    def __getitem__(self, idx):
        if idx < len(self.normal_images):
            image_path = self.normal_images[idx]
            label = 1  # 标记正常样本
        else:
            image_path = self.anomaly_images[idx - len(self.normal_images)]
            label = 0  # 标记异常样本
        
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((448, 320)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
])

# 创建数据集和数据加载器
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

# 初始化模型
generator = Generator(latent_dim).cuda()
discriminator = Discriminator().cuda()

# 初始化优化器
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# 损失函数
adversarial_loss = nn.BCELoss()

# 训练GAN
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        
        # 将数据加载到GPU
        real_imgs = imgs.cuda()
        labels = labels.cuda()

        # 标签
        valid = torch.ones(batch_size, 1).cuda()
        fake = torch.zeros(batch_size, 1).cuda()

        # -----------------
        #  训练判别器
        # -----------------
        optimizer_d.zero_grad()

        # 真实样本损失
        real_pred = discriminator(real_imgs)
        d_loss_real = adversarial_loss(real_pred, valid)

        # 生成假样本
        z = torch.randn(batch_size, latent_dim).cuda()
        fake_imgs = generator(z)
        
        # 假样本损失
        fake_pred = discriminator(fake_imgs.detach())
        d_loss_fake = adversarial_loss(fake_pred, fake)

        # 反向传播与优化
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # -----------------
        #  训练生成器
        # -----------------
        optimizer_g.zero_grad()

        # 生成器损失
        gen_pred = discriminator(fake_imgs)
        g_loss = adversarial_loss(gen_pred, valid)

        # 反向传播与优化
        g_loss.backward()
        optimizer_g.step()

    # 打印损失
    print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # 每隔一定次数保存生成图像
    if epoch % 10 == 0:
        z = torch.randn(64, latent_dim).cuda()
        gen_imgs = generator(z)
        gen_imgs = gen_imgs.detach().cpu()
        # save_image(gen_imgs, f"../output/{epoch}.png", nrow=8, normalize=True)

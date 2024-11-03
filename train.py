import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision.utils import save_image
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Загрузка конфигурационного файла JSON
with open("config.json", "r") as file:
    config = json.load(file)

# Определение датасета для загрузки BMP изображений
class BmpDataset(Dataset):
    def __init__(self, folder_path, transform=None):  # Конструктор принимает аргументы
        self.folder_path = folder_path
        self.transform = transform
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.images[idx])
        image = Image.open(img_path).convert('RGB')  # Преобразование в RGB

        if self.transform:
            image = self.transform(image)

        return image

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize(tuple(config['image_processing']['transform']['resize'])),
    transforms.ToTensor(),
    transforms.Normalize(config['image_processing']['transform']['normalize_mean'], 
                         config['image_processing']['transform']['normalize_std'])
])

# Инициализация датасета и загрузчика данных
dataset = BmpDataset(config['paths']['dataset_folder'], transform)
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

# Определение модели генератора
class GeneratorHippocampus(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(GeneratorHippocampus, self).__init__()

        def block(input_dim, output_dim, kernel_size=4, stride=2, padding=1, normalize=True):
            layers = [nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding)]
            if normalize:
                layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 512, kernel_size=4, stride=1, padding=0, normalize=False),  # Output 4x4
            *block(512, 256),  # Output 8x8
            *block(256, 128),  # Output 16x16
            *block(128, 64),   # Output 32x32
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),  # Output 64x64
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # Стартовое скрытое состояние [latent_dim, 1, 1]
        img = self.model(z)
        return img

# Определение модели дискриминатора
class DiscriminatorHippocampus(nn.Module):
    def __init__(self, img_size, channels):
        super(DiscriminatorHippocampus, self).__init__()

        def block(input_dim, output_dim, kernel_size=4, stride=2, padding=1, normalize=True):
            layers = [nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)]
            if normalize:
                layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(channels, 64, normalize=False),  # Output 32x32
            *block(64, 128),  # Output 16x16
            *block(128, 256),  # Output 8x8
            *block(256, 512),  # Output 4x4
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # Output 1x1 (для дискриминатора)
        )

    def forward(self, img):
        return self.model(img).view(-1)

# Функция для вычисления градиентной пенализации
def compute_gradient_penalty(D, real_samples, fake_samples):
    # Приведение fake_samples к размеру real_samples
    fake_samples = torch.nn.functional.interpolate(fake_samples, size=real_samples.size()[2:])
    
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(real_samples.device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Основная функция для обучения WGAN-GP
def train_hippocampus(dataloader, generator, discriminator, latent_dim, epochs, n_critic, lambda_gp, lr, betas):
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    os.makedirs(config['paths']['output_folder'], exist_ok=True)
    os.makedirs(config['paths']['model_save_folder'], exist_ok=True)

    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            real_imgs = imgs.cuda()
            batch_size = real_imgs.size(0)

            # --- Тренировка дискриминатора ---
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim).cuda()
            fake_imgs = generator(z)
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs)) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # --- Тренировка генератора каждые n_critic шагов ---
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, latent_dim).cuda()
                fake_imgs = generator(z)
                g_loss = -torch.mean(discriminator(fake_imgs))
                g_loss.backward()
                optimizer_G.step()

            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

        # Сохранение изображений через каждые 100 эпох
        if epoch % 100 == 0:
            save_image(fake_imgs.data[:25], f"{config['paths']['output_folder']}/generated_images_{epoch}.png", nrow=5, normalize=True)

        # Сохранение моделей через каждые 1000 эпох
        if epoch % 1000 == 0:
            torch.save(generator.state_dict(), f"{config['paths']['model_save_folder']}/generator_hippocampus_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{config['paths']['model_save_folder']}/discriminator_hippocampus_epoch_{epoch}.pth")

    # Сохранение финальных моделей
    torch.save(generator.state_dict(), f"{config['paths']['model_save_folder']}/generator_hippocampus_final.pth")
    torch.save(discriminator.state_dict(), f"{config['paths']['model_save_folder']}/discriminator_hippocampus_final.pth")

# Параметры обучения из конфигурации
latent_dim = config['model']['latent_dim']
img_size = config['model']['img_size']
channels = config['model']['channels']
epochs = config['training']['epochs']
n_critic = config['training']['n_critic']
lambda_gp = config['training']['lambda_gp']
lr = config['training']['lr']
betas = tuple(config['training']['betas'])

# Инициализация моделей
generator = GeneratorHippocampus(latent_dim, img_size, channels).cuda()
discriminator = DiscriminatorHippocampus(img_size, channels).cuda()

# Запуск обучения модели "Гиппокамп"
train_hippocampus(dataloader, generator, discriminator, latent_dim, epochs, n_critic, lambda_gp, lr, betas)

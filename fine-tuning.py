import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import json

# Функции для сохранения и загрузки модели
def save_model(generator, discriminator, epoch, save_dir="models"):
    torch.save(generator.state_dict(), f"{save_dir}/generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_epoch_{epoch}.pth")

def load_model(generator, discriminator, generator_path, discriminator_path):
    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discriminator_path))

# Функция для дообучивания
def continue_training(dataloader, generator, discriminator, latent_dim, epochs, n_critic, lambda_gp, lr, betas, start_epoch, save_dir="models"):
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    
    for epoch in range(start_epoch, start_epoch + epochs):
        for i, imgs in enumerate(dataloader):
            real_imgs = imgs.cuda()
            batch_size = real_imgs.size(0)

            # --- Обучение дискриминатора ---
            optimizer_D.zero_grad()

            # Генерация случайного шума
            z = torch.randn(batch_size, latent_dim).cuda()

            # Генерация фейковых изображений
            fake_imgs = generator(z)

            # Вычисление градиентной пенализации
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

            # Рассчитываем потери дискриминатора
            d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs)) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # --- Обучение генератора каждые n_critic шагов ---
            if i % n_critic == 0:
                optimizer_G.zero_grad()

                # Повторно генерируем фейковые изображения
                fake_imgs = generator(z)

                # Рассчитываем потери генератора
                g_loss = -torch.mean(discriminator(fake_imgs))
                g_loss.backward()
                optimizer_G.step()

            print(f"[Epoch {epoch}/{start_epoch + epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Сохраняем сгенерированные изображения для визуализации прогресса
        if epoch % 100 == 0:
            save_image(fake_imgs.data[:25], f"{save_dir}/generated_images_epoch_{epoch}.png", nrow=5, normalize=True)

        # Сохраняем модель каждые 1000 эпох
        if epoch % 1000 == 0:
            save_model(generator, discriminator, epoch, save_dir)

    # Сохранение финальных весов после дообучения
    save_model(generator, discriminator, start_epoch + epochs - 1, save_dir)

# Загрузка config.json
with open('fine-config.json', 'r') as f:
    config = json.load(f)

# Параметры модели и обучения
latent_dim = config['training']['latent_dim']
epochs = config['training']['epochs']
n_critic = config['training']['n_critic']
lambda_gp = config['training']['lambda_gp']
lr = config['training']['lr']
betas = tuple(config['training']['betas'])
start_epoch = config['training']['start_epoch']
batch_size = config['training']['batch_size']

# Пути к весам и данным
dataset_folder = config['paths']['dataset_folder']
generator_path = config['paths']['generator_weights']
discriminator_path = config['paths']['discriminator_weights']
save_dir = config['paths']['save_dir']

# Параметры изображения
img_size = config['model']['img_size']
channels = config['model']['channels']

# Инициализация датасета и загрузчика данных
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

dataset = BmpDataset(dataset_folder, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Инициализация генератора и дискриминатора
generator = GeneratorHippocampus(latent_dim, img_size, channels).cuda()
discriminator = DiscriminatorHippocampus(img_size, channels).cuda()

# Загрузка сохранённых весов модели
load_model(generator, discriminator, generator_path, discriminator_path)

# Продолжаем обучение модели "Гиппокамп"
continue_training(dataloader, generator, discriminator, latent_dim, epochs, n_critic, lambda_gp, lr, betas, start_epoch, save_dir)

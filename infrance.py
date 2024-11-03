from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torchvision.utils import save_image
from io import BytesIO
from fastapi.responses import StreamingResponse
from PIL import Image
import uvicorn
from pyngrok import ngrok

# Загрузка сохранённой модели
class GeneratorHippocampus(torch.nn.Module):
    # Определение модели (копия архитектуры)
    def _init_(self, latent_dim, img_size, channels):
        super(GeneratorHippocampus, self)._init_()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        # Определите здесь слои так же, как в вашем генераторе
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, channels, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.model(z)
        return img

# Настройки FastAPI
app = FastAPI()

# Пути к весам и параметры модели
generator_path = "generator_final.pth"  # Обязательно загрузите сюда свои веса
latent_dim = 100
img_size = 64
channels = 3

# Инициализация генератора и загрузка весов
generator = GeneratorHippocampus(latent_dim, img_size, channels)
generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
generator.eval()

# Модель запроса для генерации изображений
class GenerateRequest(BaseModel):
    num_images: int = 1  # Количество изображений для генерации

# Функция для генерации изображений
def generate_images(generator, latent_dim, num_images):
    z = torch.randn(num_images, latent_dim)
    with torch.no_grad():
        generated_images = generator(z)
    return generated_images

# Эндпоинт для генерации изображений
@app.post("/generate/")
async def generate(request: GenerateRequest):
    num_images = request.num_images

    # Генерация изображений
    images_tensor = generate_images(generator, latent_dim, num_images)

    # Преобразуем изображения в формат PIL и отправляем пользователю
    save_image(images_tensor, "temp.png", nrow=num_images, normalize=True)
    img = Image.open("temp.png")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")

# Настройка ngrok и запуск приложения
ngrok_tunnel = ngrok.connect(8000)
print("Public URL:", ngrok_tunnel.public_url)

# Запуск FastAPI через uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)

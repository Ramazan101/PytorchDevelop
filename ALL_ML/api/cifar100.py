from fastapi import APIRouter, UploadFile, File
import io, torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Linear(256, 100)
    )

  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

classess = ['apple',
 'aquarium_fish',
 'baby',
 'bear',
 'beaver',
 'bed',
 'bee',
 'beetle',
 'bicycle',
 'bottle',
 'bowl',
 'boy',
 'bridge',
 'bus',
 'butterfly',
 'camel',
 'can',
 'castle',
 'caterpillar',
 'cattle',
 'chair',
 'chimpanzee',
 'clock',
 'cloud',
 'cockroach',
 'couch',
 'crab',
 'crocodile',
 'cup',
 'dinosaur',
 'dolphin',
 'elephant',
 'flatfish',
 'forest',
 'fox',
 'girl',
 'hamster',
 'house',
 'kangaroo',
 'keyboard',
 'lamp',
 'lawn_mower',
 'leopard',
 'lion',
 'lizard',
 'lobster',
 'man',
 'maple_tree',
 'motorcycle',
 'mountain',
 'mouse',
 'mushroom',
 'oak_tree',
 'orange',
 'orchid',
 'otter',
 'palm_tree',
 'pear',
 'pickup_truck',
 'pine_tree',
 'plain',
 'plate',
 'poppy',
 'porcupine',
 'possum',
 'rabbit',
 'raccoon',
 'ray',
 'road',
 'rocket',
 'rose',
 'sea',
 'seal',
 'shark',
 'shrew',
 'skunk',
 'skyscraper',
 'snail',
 'snake',
 'spider',
 'squirrel',
 'streetcar',
 'sunflower',
 'sweet_pepper',
 'table',
 'tank',
 'telephone',
 'television',
 'tiger',
 'tractor',
 'train',
 'trout',
 'tulip',
 'turtle',
 'wardrobe',
 'whale',
 'willow_tree',
 'wolf',
 'woman',
 'worm']


chech_cifar100 = APIRouter(prefix='/cifar100', tags=['Cifar100class'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage().to(device)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'all_models'
model_path = MODEL_DIR / 'model_cifar.pth'

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


@chech_cifar100.post('/cifar100')
async def cifar100(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(image_tensor)
        print(image_tensor.shape)
        pred = y.argmax(dim=1).item()

    return {
        'image_this' : classess[pred]
    }
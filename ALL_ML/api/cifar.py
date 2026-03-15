from fastapi import  APIRouter, UploadFile, File, HTTPException
import io, torch
from torchvision import transforms
from PIL import Image
from ALL_ML.database.module import CofarClassification3
from pathlib import Path

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

classes = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

check_rain_image = APIRouter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CofarClassification3().to(device)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'all_models'
model_path = MODEL_DIR / 'model_cifar.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()



@check_rain_image.post('/cifar')
async def cifar_cloth(image: UploadFile = File(...)):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(image_tensor)
        print(image_tensor.shape)
        prediction = y.argmax(dim=1)[0].item()

    return {'image_name': classes[prediction]}

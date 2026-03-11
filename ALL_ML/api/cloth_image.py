from fastapi import  APIRouter, UploadFile, File, HTTPException
import io, torch
from torchvision import transforms
from PIL import Image
from ALL_ML.database.module import CheckCloths
from pathlib import Path

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',

]

check_cloths = APIRouter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckCloths()
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'all_models'
model_path = MODEL_DIR / 'model_testing.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@check_cloths.post('/predict_image')
async def predict_cloth(image: UploadFile = File(...)):
    try:
        image_data = await image.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='No image')
        img = Image.open(io.BytesIO(image_data))
        image_tenso = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(image_tenso)
            pred = y.argmax(dim=1).item()
        cloth_name = classes[pred]
        return {'Answer': pred,
                'class': cloth_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
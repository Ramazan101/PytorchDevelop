from fastapi import APIRouter, UploadFile, File, HTTPException
import io, torch
from pathlib import Path
from torchvision import transforms
from ALL_ML.database.module import CheckImage
from PIL import Image


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

check_image_app = APIRouter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'all_models'
model_path = MODEL_DIR / 'model.pth'

model = CheckImage()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


@check_image_app.post('/predict_numbers')
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=404, detail="No image")
        img = Image.open(io.BytesIO(image_data))
        image_tenso = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(image_tenso)
            pred = y_pred.argmax(dim=1).item()
        return {'Answer': pred}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





from fastapi import APIRouter, UploadFile, File, HTTPException
import io, torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from ALL_ML.database.module import ChechImage


train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])



check_transport = APIRouter(prefix='/transport', tags=['Transport'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'all_models'
model_path = MODEL_DIR / 'model_transport.pth'

model = ChechImage()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
classes = ['airplane', 'atrain', 'ship', 'tank']


@check_transport.post('/check_transports')
async def check_flower(image: UploadFile = File(...)):
    try:
        flower = await image.read()
        if not flower:
            raise HTTPException(status_code=400, detail='Image is empty')
        img = Image.open(io.BytesIO(flower)).convert('RGB')
        image_tensor = test_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y_predict = model(image_tensor)
            prediction = torch.argmax(y_predict, dim=1).item()
        return {'this class' : classes[prediction]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



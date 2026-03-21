from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms, models

router = APIRouter(prefix='/food', tags=['Food'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ALL_ML/all_models/food_model.pth"

# 1. Загружаем checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]

# 2. Создаем ту же архитектуру, что была при обучении
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

# 3. Загружаем веса
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# 4. transform как при test/predict
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@router.post("/check_food")
async def check_food(image: UploadFile = File(...)):
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        predicted_class = class_names[pred.item()]
        confidence = round(conf.item() * 100, 2)

        return {
            "this class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI, UploadFile, File, HTTPException
import io, torch
import uvicorn
from torchvision import datasets, transforms
import torch.nn as nn
from PIL import Image

class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16 * 14 * 14, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

check_image_app = FastAPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CheckImage()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

@check_image_app.post('/predict')
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


if __name__ == "__main__":
    uvicorn.run(check_image_app, host="127.0.0.1", port=8000)



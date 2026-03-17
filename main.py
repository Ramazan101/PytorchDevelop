from fastapi import FastAPI
import uvicorn
from ALL_ML.api import (numbers_image, cloth_image, cifar, flowers_image)

ALL_ML = FastAPI()

ALL_ML.include_router(numbers_image.check_image_app)
ALL_ML.include_router(cloth_image.check_cloths)
ALL_ML.include_router(cifar.check_rain_image)
ALL_ML.include_router(flowers_image.check_flowers)

if __name__ == '__main__':
    uvicorn.run(ALL_ML, host='127.0.0.1', port=8000)
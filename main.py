from fastapi import FastAPI
import uvicorn
from ALL_ML.api import (numbers_image, cloth_image, cifar, flowers_image, smartphone, food_image, transports)

ALL_ML = FastAPI()

ALL_ML.include_router(numbers_image.check_image_app)
ALL_ML.include_router(cloth_image.check_cloths)
ALL_ML.include_router(cifar.check_rain_image)
ALL_ML.include_router(flowers_image.check_flowers)
ALL_ML.include_router(smartphone.check_smartphone)
ALL_ML.include_router(food_image.router)
ALL_ML.include_router(transports.check_transport)

if __name__ == '__main__':
    uvicorn.run(ALL_ML, host='127.0.0.1', port=8000)
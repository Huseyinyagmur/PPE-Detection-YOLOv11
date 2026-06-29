from ultralytics import YOLO
from config import (
    MODEL_NAME,
    DATA_YAML,
    EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE,
    PROJECT_NAME,
    RUN_NAME,
    CACHE,
    WORKERS
)
if __name__ == "__main__":
    model=YOLO(MODEL_NAME)

    model.train(
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        workers=WORKERS,
        cache=CACHE
    )
from ultralytics import YOLO
from config import BEST_MODEL_PATH,SOURCE_IMAGE,CONF_THRESHOLD

model=YOLO(BEST_MODEL_PATH)

results=model.predict(
    source=SOURCE_IMAGE,
    conf=CONF_THRESHOLD,
    save=True,
)
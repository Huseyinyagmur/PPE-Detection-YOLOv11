from ultralytics import YOLO
from config import WEBCAM_SOURCE,BEST_MODEL_PATH,CONF_THRESHOLD

model=YOLO(BEST_MODEL_PATH)

model.predict(
    source=WEBCAM_SOURCE,
    conf=CONF_THRESHOLD,
    show=True
)

MODEL_NAME = r"C:\Users\husey\runs\detect\.runs\yolo11s_ppe-2\weights\last.pt"

DATA_YAML="data.yaml"

EPOCHS=50

IMAGE_SIZE=640

BATCH_SIZE=8

DEVICE=0

PROJECT_NAME = ".runs"

RUN_NAME = "yolo11s_ppe"

WORKERS = 0
CACHE = False

BEST_MODEL_PATH = "models/best.pt"

SOURCE_IMAGE = "dataset/css-data/test/images"

CONF_THRESHOLD = 0.25

WEBCAM_SOURCE=0
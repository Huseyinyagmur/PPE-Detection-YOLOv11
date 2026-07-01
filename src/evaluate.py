from ultralytics import YOLO
from config import BEST_MODEL_PATH,DATA_YAML,IMAGE_SIZE,BATCH_SIZE,DEVICE


if __name__=="__main__":
    model=YOLO(BEST_MODEL_PATH)

    metrics=model.val(
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE
    )
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
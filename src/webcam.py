from ultralytics import YOLO
from config import WEBCAM_SOURCE,BEST_MODEL_PATH,CONF_THRESHOLD,WINDOW_NAME
import cv2
import time

model=YOLO(BEST_MODEL_PATH)
cap=cv2.VideoCapture(WEBCAM_SOURCE)

if not cap.isOpened():
    print("Camere coult not be opened")
    exit()
    
while True:
    ret,frame=cap.read()

    if not ret:
        print("Frame could not be read")
        break
    results=model(frame,conf=CONF_THRESHOLD)
    annotated_frame=results[0].plot()
    cv2.imshow(WINDOW_NAME,annotated_frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
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
    start_time=time.time()
    ret,frame=cap.read()

    if not ret:
        print("Frame could not be read")
        break
    results=model(frame,conf=CONF_THRESHOLD)
    annotated_frame=results[0].plot()
    end_time=time.time()
    fps=1/(end_time-start_time)
    cv2.putText(annotated_frame,f"FPS:{fps:.2f}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow(WINDOW_NAME,annotated_frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
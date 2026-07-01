# 🦺 PPE Detection with YOLOv11

A deep learning-based Personal Protective Equipment (PPE) detection system developed using **YOLOv11** and **Ultralytics**. The model detects multiple safety-related objects in construction environments, helping improve workplace safety through real-time computer vision.

---

# 📌 Project Overview

This project focuses on detecting workers and their personal protective equipment (PPE) using an object detection model trained on a construction site safety dataset.

The system detects the following classes:

- Hardhat
- Mask
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest
- Machinery
- Vehicle

The project is built using **Ultralytics YOLOv11**, leveraging transfer learning from pretrained weights to achieve accurate and efficient object detection.

---

# 🚀 Features

- YOLOv11 object detection
- Transfer Learning
- GPU (CUDA) support
- Modular project architecture
- Training pipeline
- Image inference
- Model evaluation
- YOLO11n vs YOLO11s comparison
- Automatic checkpoint saving

---

# 📂 Project Structure

```text
PPE-Detection-YOLOv11/
│
├── dataset/
├── models/
│   └── best.pt
│
├── runs/
├── results/
│
├── src/
│   ├── config.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── webcam.py
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies

- Python
- PyTorch
- Ultralytics YOLOv11
- OpenCV
- NumPy
- Matplotlib

---

# 📊 Model Performance

### YOLO11s Validation Results

| Metric | Score |
|---------|------:|
| Precision | **0.9015** |
| Recall | **0.7563** |
| mAP@50 | **0.8222** |
| mAP@50-95 | **0.5232** |

---

# 📈 Model Comparison

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|----------:|-------:|--------:|-----------:|
| YOLO11n | 0.870 | 0.709 | 0.765 | 0.456 |
| **YOLO11s** | **0.902** | **0.756** | **0.822** | **0.523** |

YOLO11s achieved higher detection accuracy than YOLO11n while maintaining real-time inference performance.

---

# 🖼️ Results

## Training Results

> Training curves and evaluation metrics will be added here.

## Detection Examples

> Sample inference images will be added here.

---

# ▶️ Training

```bash
python src/train.py
```

---

# 🔍 Inference

```bash
python src/predict.py
```

---

# 📊 Evaluation

```bash
python src/evaluate.py
```

---

# 🎯 Future Improvements

- Real-time webcam detection
- Video inference
- ONNX export
- TensorRT optimization
- FPS benchmark
- Latency analysis
- Docker deployment
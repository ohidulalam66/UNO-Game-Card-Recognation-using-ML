# UNO Game Card Recognition using YOLOv8

## Overview
This project trains a YOLOv8 object detection model to recognize UNO game cards from images. The notebook covers the complete workflow:

- Dataset extraction and preparation
- Automatic generation of `data.yaml`
- YOLOv8 training using Ultralytics
- Saving trained weights to Google Drive
- Loading the trained model
- Running inference on uploaded images
- Visualizing detection results

## Notebook Workflow

### 1. Connect Google Drive
The notebook mounts Google Drive to:
- Access datasets
- Save training outputs
- Store trained model weights

### 2. Install Dependencies
The project uses:

```bash
pip install ultralytics
```

Main libraries:
- Ultralytics YOLOv8
- OpenCV
- Matplotlib
- Google Colab utilities

### 3. Dataset Preparation
The notebook:

1. Extracts a ZIP dataset.
2. Locates image and label directories.
3. Loads class names from `classes.txt`.
4. Creates a YOLO-compatible `data.yaml` configuration file.

Expected dataset structure:

```text
dataset/
├── images/
├── labels/
└── classes.txt
```

### 4. Model Training

Training configuration:

| Parameter | Value |
|-----------|--------|
| Model | YOLOv8 Nano (`yolov8n.pt`) |
| Epochs | 120 |
| Image Size | 640 |
| Batch Size | 8 |
| Patience | 25 |

Example:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=120,
    imgsz=640,
    batch=8,
    patience=25
)
```

### 5. Saving Model Weights

The notebook saves:
- `best.pt` (best-performing model)
- `last.pt` (latest checkpoint)

to Google Drive for future use.

### 6. Model Loading

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

### 7. UNO Card Detection

The trained model can perform inference on uploaded images:

```python
results = model.predict(
    source="image.jpg",
    conf=0.25
)
```

Features:
- Upload image through Google Colab
- Detect UNO cards
- Display bounding boxes
- Show predicted card classes
- Print detection details

## Requirements

```text
ultralytics
opencv-python
matplotlib
```

## Running the Project

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Upload or connect the UNO dataset.
4. Run all cells sequentially.
5. Train the model.
6. Save the generated weights.
7. Upload test images and perform inference.

## Output

The trained model produces:

- YOLOv8 weights (`best.pt`, `last.pt`)
- Training logs and metrics
- UNO card detection results on custom images

## Project Goal

Build an object detection model capable of recognizing UNO cards from images using YOLOv8, enabling automated card identification for computer vision applications and game-related projects.

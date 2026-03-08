# YOLOE Object Detection on Mac - Complete Guide

### used this outdated video to download yolo but had to expand to yoloe using some raspberry pi stuff on yoloe as i couldnt find any other helpful stuff and deepseek was way more help than chatgpt

https://youtu.be/s5xdH9aluds?si=KgL5eeSWpCaE9vaK

https://core-electronics.com.au/guides/raspberry-pi/custom-object-detection-models-without-training-yoloe-and-raspberry-pi/

crack3deepsetclass -- shows you must set class using YOLOE

macyoloenp -- no prompt YOLOE will built in classes

yolo11ntest -- earlier less good yolon

## 📋 Overview

This guide provides comprehensive documentation for running YOLOE (YOLO with Embeddings) object detection on macOS using a webcam. YOLOE is a powerful vision model that accepts **text prompts** to detect custom objects without requiring training data.

## 📦 Prerequisites

### Hardware Requirements

- Mac computer (Intel or Apple Silicon)
- Webcam (built-in or external)

### Software Requirements

- Python 3.8 or higher
- pip package manager
- macOS 11.0 or higher

## 🔧 Installation

### 1. Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python3 -m venv yoloe-env
source yoloe-env/bin/activate

# Or use conda if you prefer
conda create --name yoloe-env python=3.11 -y
conda activate yoloe-env
```

### 2. Install Required Packages

```bash
# Install Ultralytics (includes YOLOE)
pip install ultralytics

# Install OpenCV for webcam access
pip install opencv-python

# Verify installation
python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"
```

### 3. Download YOLOE Model

The model will automatically download on first use, or you can pre-download:

```python
from ultralytics import YOLOE

# This will download the model automatically
model = YOLOE("yoloe-11s-seg.pt")
print("Model downloaded successfully!")
```

## 🚀 Basic Usage: Simple Object Detection

### Basic Webcam Detection

```python
import cv2
from ultralytics import YOLOE

# Load YOLOE model
model = YOLOE("yoloe-11s-seg.pt")

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define what to detect
model.set_classes(["person", "car", "dog", "cat"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(frame, conf=0.25)

    # Show results (boxes only, no masks)
    annotated = results[0].plot(boxes=True, masks=False)
    cv2.imshow("YOLOE Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🎯 Custom Object Detection with Text Prompts

### Detecting Pavement Cracks

```python
import cv2
from ultralytics import YOLOE
import time

# Load model
model = YOLOE("yoloe-11s-seg.pt")

# Define crack detection prompts
crack_prompts = [
    "pavement crack",
    "road fissure",
    "concrete fracture",
    "asphalt crack",
    "thin dark line on road",
    "jagged split in pavement",
    "hairline crack",           # Very thin cracks
    "transverse crack",         # Cracks across the road
    "longitudinal crack",       # Cracks along the road
    "alligator crack",          # Network of small cracks
    "block crack",              # Rectangular crack patterns
    "edge crack"                # Cracks near road edge
]

# Set the classes
model.set_classes(crack_prompts)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting crack detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference with lower confidence for faint cracks
    results = model.predict(frame, conf=0.15, verbose=False)

    # Display with boxes only (no masks for better performance)
    annotated = results[0].plot(boxes=True, masks=False)

    # Add FPS counter
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time if inference_time > 0 else 0
    cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show detection count
    num_detections = len(results[0].boxes)
    cv2.putText(annotated, f'Cracks: {num_detections}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Crack Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🎨 Advanced: Multi-Class Detection with Custom Labels

### Organizing Multiple Object Categories

```python
import cv2
from ultralytics import YOLOE
import numpy as np

# Load model
model = YOLOE("yoloe-11s-seg.pt")

# Define multiple categories with their prompts
categories = {
    "crack": [
        "pavement crack",
        "road fissure",
        "concrete fracture",
        "asphalt crack",
        "hairline crack"
    ],
    "pothole": [
        "pothole",
        "road depression",
        "hole in pavement",
        "broken road surface"
    ],
    "patch": [
        "road patch",
        "asphalt repair",
        "concrete patch",
        "fixed road section"
    ],
    "manhole": [
        "manhole cover",
        "utility access",
        "metal circle on road"
    ]
}

# Flatten all prompts for model
all_prompts = []
for category, prompts in categories.items():
    all_prompts.extend(prompts)

# Create mapping from prompt index to category
prompt_to_category = {}
for category, prompts in categories.items():
    for prompt in prompts:
        prompt_to_category[prompt] = category

# Set all classes
model.set_classes(all_prompts)

# Color mapping for display
category_colors = {
    "crack": (0, 255, 0),      # Green
    "pothole": (0, 0, 255),    # Red
    "patch": (255, 165, 0),    # Orange
    "manhole": (255, 0, 255),  # Purple
    "unknown": (255, 255, 255) # White
}

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(frame, conf=0.15, verbose=False)

    # Custom drawing for better control
    annotated = frame.copy()

    if len(results[0].boxes) > 0:
        boxes = results[0].boxes

        for box, class_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            prompt = all_prompts[int(class_id)]
            category = prompt_to_category.get(prompt, "unknown")
            color = category_colors.get(category, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label with category
            label = f"{category} ({conf:.2f})"
            cv2.putText(annotated, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add instructions
    cv2.putText(annotated, "Press 'q' to quit | 's' to save", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Multi-Class Detection", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Save screenshot
        cv2.imwrite(f"detection_{time.time()}.jpg", annotated)

cap.release()
cv2.destroyAllWindows()
```

## 📊 Understanding the Code Components

### Model Loading

```python
from ultralytics import YOLOE  # Import the specific YOLOE class

# Load different model sizes
model = YOLOE("yoloe-11s-seg.pt")   # Small model (faster)
model = YOLOE("yoloe-11m-seg.pt")   # Medium model (balanced)
model = YOLOE("yoloe-11l-seg.pt")   # Large model (more accurate)
```

### Setting Classes (Critical Step!)

```python
# THIS IS ESSENTIAL - tell the model what to look for
model.set_classes(["pavement crack", "road fissure", "concrete fracture"])

# The model will now only detect these objects
# The prompts can be descriptive phrases, not just single words
```

### Running Inference

```python
# Basic inference
results = model.predict(frame)

# With custom parameters
results = model.predict(
    frame,
    conf=0.15,        # Confidence threshold (lower = more detections)
    iou=0.5,          # Intersection over Union threshold
    verbose=False,    # Suppress output
    device='cpu'      # Use CPU (or 'mps' for Apple Silicon)
)
```

### Display Options

```python
# Plot() controls what's shown
annotated = results[0].plot(
    boxes=True,      # Show bounding boxes
    masks=False,     # Hide segmentation masks (faster)
    conf=True,       # Show confidence scores
    labels=True      # Show class labels
)

# Boxes only (best performance)
annotated = results[0].plot(boxes=True, masks=False)

# Boxes + masks (slower, more detailed)
annotated = results[0].plot(boxes=True, masks=True)
```

## ⚙️ Configuration Options

### Confidence Threshold

```python
# Lower threshold = more detections (but more false positives)
results = model.predict(frame, conf=0.1)   # Very sensitive
results = model.predict(frame, conf=0.25)  # Default
results = model.predict(frame, conf=0.5)   # More confident only
```

### Resolution Settings

```python
# Lower resolution = faster but less accurate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Higher resolution = slower but more detailed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

## 🚀 Performance Optimization for Mac

### For Apple Silicon (M1/M2/M3)

```python
# Use MPS acceleration (Metal Performance Shaders)
results = model.predict(frame, device='mps')
```

### Frame Skipping

```python
# Process every other frame to increase FPS
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count += 1

    if frame_count % 2 == 0:  # Process every 2nd frame
        results = model.predict(frame)
        # ... display code
```

## 📝 Saving Results

```python
import csv
from datetime import datetime

def save_detections(results, frame_number):
    """Save detection results to CSV file"""

    filename = f"detections_{datetime.now().strftime('%Y%m%d')}.csv"

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file is new
        if f.tell() == 0:
            writer.writerow(['frame', 'timestamp', 'class_id', 'class_name',
                           'confidence', 'x1', 'y1', 'x2', 'y2'])

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box, class_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                writer.writerow([
                    frame_number,
                    datetime.now().isoformat(),
                    int(class_id),
                    crack_prompts[int(class_id)],
                    f"{float(conf):.3f}",
                    *[int(x) for x in box[:4]]
                ])
```

## 🐛 Troubleshooting

### Common Issues and Solutions

| Issue                           | Solution                                                    |
| ------------------------------- | ----------------------------------------------------------- |
| `No module named 'ultralytics'` | Run `pip install ultralytics`                               |
| Webcam not opening              | Check `ls /dev/video*` or try `cap = cv2.VideoCapture(1)`   |
| No detections                   | Lower confidence: `conf=0.1` or improve prompts             |
| Slow performance                | Lower resolution, use `masks=False`, try `yoloe-11n-seg.pt` |
| Model download fails            | Manual download from Ultralytics GitHub                     |
| `mobileclip_blt.ts` not found   | Will auto-download on first use                             |

### Verify Installation

```python
python -c "
from ultralytics import YOLOE
model = YOLOE('yoloe-11s-seg.pt')
print('✅ YOLOE loaded successfully')
print(f'Model type: {type(model)}')
print(f'Set classes method: {hasattr(model, \"set_classes\")}')
"
```

## 📚 Additional Resources

- [Ultralytics YOLOE Documentation](https://docs.ultralytics.com/models/yoloe/)
- [YOLOE Research Paper](https://arxiv.org/abs/2411.17605)
- [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)

## 🔄 Quick Start Template

```python
import cv2
from ultralytics import YOLOE

# Configuration
MODEL_NAME = "yoloe-11s-seg.pt"
CLASSES = ["pavement crack", "road fissure", "concrete fracture"]
CONFIDENCE = 0.15
RESOLUTION = (640, 480)

# Initialize
model = YOLOE(MODEL_NAME)
model.set_classes(CLASSES)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

print(f"Detecting: {CLASSES}")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONFIDENCE, verbose=False)
    annotated = results[0].plot(boxes=True, masks=False)

    cv2.imshow("YOLOE Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📌 Important Notes

1. **First run will download models** (~50-100MB each)
2. **YOLOE requires `set_classes()`** before inference
3. **Use descriptive prompts** for better detection
4. **Lower confidence** (0.1-0.15) for faint objects
5. **`masks=False`** significantly improves performance

Happy detecting! 🎉

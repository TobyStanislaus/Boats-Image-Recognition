# Boat Image Recognition

A computer vision system for detecting and recognising different types of sailing boats from images.

Developed as the artefact for an Extended Project Qualification (EPQ), this project uses **YOLO object detection** to identify boats and classify them into different classes. It also includes functionality for recognising boat numbers using OCR.

## Overview

The system takes an image containing one or more boats and uses a trained YOLO model to:

* Detect boats within the image
* Identify the type of each boat
* Locate boat numbers
* Process multiple images automatically
* Optionally process images from a webcam
* Display detected objects and confidence scores

The project was developed around a custom dataset of sailing boats and trained using the Ultralytics YOLO framework.

## Computer Vision Pipeline

```text
                    Input Image
                         │
                         ▼
                ┌─────────────────┐
                │  YOLO Detector  │
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Boat Detection        Boat Number
              │                  Detection
              ▼                     │
      Boat Classification          ▼
              │                  OCR
              │                     │
              └──────────┬──────────┘
                         ▼
                  Final Detections
```

The model can distinguish between different boat classes while simultaneously locating the objects within an image.

## Boat Classes

The training configuration contains the following classes:

| ID | Class             |
| -: | ----------------- |
|  0 | Boat              |
|  1 | Boat Number       |
|  2 | Optimist          |
|  3 | Topper            |
|  4 | Laser             |
|  5 | Tera              |
|  6 | Feva              |
|  7 | 29er              |
|  8 | RS Double Handers |
|  9 | Aero              |
| 10 | Europe            |
| 11 | 400s              |
| 12 | Solo              |

This allows the detector to distinguish between visually similar sailing dinghies rather than simply identifying everything as a generic boat.

## Model

The project uses **Ultralytics YOLOv8 Nano** as the base model.

Training is performed using transfer learning from the pretrained `yolov8n.pt` model. The training configuration uses:

* **Model:** YOLOv8n
* **Epochs:** 300 maximum
* **Batch size:** 16
* **Early stopping patience:** 20 epochs

The training script is implemented in `train_model.py`.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="config.yaml",
    epochs=300,
    batch=16,
    patience=20
)
```

## Dataset

The dataset is structured for use with Ultralytics YOLO and contains separate training and validation image directories. The dataset configuration is defined in `config.yaml`.

```text
data/
├── images/
│   ├── ...
│
└── testing/
    ├── ...
```

Images were annotated with bounding boxes corresponding to the boat classes.

The dataset was developed specifically for this project rather than relying solely on a pre-existing generic object-detection dataset.

## Detection

`main.py` provides the main entry point for running inference.

The current configuration processes images from:

```text
./data/testing
```

using the trained model `train23`, with a confidence threshold of `0.1`.

```python
test_images_path = "./data/testing"
model_name = "train23"
threshold = 0.1
```

The system can also be configured to use a webcam as an input source.

## OCR

Boat numbers are treated as a separate detection class so that they can be located within an image before being passed to an OCR system.

The project uses **EasyOCR** for text recognition.

This enables the system to go beyond identifying the type of boat and extract information such as its sail number.

## Installation

### Requirements

* Python 3.10–3.11
* Ultralytics
* EasyOCR

The original project specifies Python 3.10–3.11 as its supported versions.

Clone the repository:

```bash
git clone https://github.com/TobyStanislaus/Boats-Image-Recognition.git
cd Boats-Image-Recognition
```

Install the required packages:

```bash
pip install ultralytics
pip install easyocr
```

## Training

To train a new model, configure the dataset in:

```text
config.yaml
```

Then run:

```bash
python train_model.py
```

The script loads the pretrained YOLOv8 Nano model and trains it on the custom dataset.

Training results are saved by Ultralytics in the `runs/detect` directory.

## Running Inference

Once a model has been trained, configure the model name and testing directory in `main.py`.

Then run:

```bash
python main.py
```

The program processes the test images and reports the total inference time.

Detection visualisation can be enabled through:

```python
showImg = True
showCrop = True
```

and webcam input can be enabled with:

```python
useWebCam = True
```

## Project Structure

```text
Boats-Image-Recognition/
│
├── data/
│   ├── images/              # Training/validation images
│   └── testing/             # Images used for inference
│
├── database/                # Project database/data files
├── runs/
│   └── detect/              # YOLO training and inference results
│
├── toolScripts/             # Supporting scripts
│
├── main.py                  # Main inference program
├── train_model.py           # YOLO model training
├── tools.py                 # Detection, OCR and processing utilities
├── config.yaml              # Dataset and class configuration
│
├── yolov8n.pt               # YOLOv8 Nano pretrained model
├── yolo11n.pt               # YOLO11 Nano model
│
├── race_data.txt            # Race-related data
└── LICENSE
```

## Technologies

* **Python**
* **Ultralytics YOLO**
* **YOLOv8**
* **OpenCV**
* **EasyOCR**
* **Object Detection**
* **Computer Vision**
* **Transfer Learning**

## Key Learning Outcomes

This project provided practical experience with the complete machine-learning workflow:

1. **Dataset creation** — collecting and annotating domain-specific images.
2. **Object detection** — training YOLO to locate and classify objects.
3. **Transfer learning** — fine-tuning a pretrained neural network on a custom dataset.
4. **Model evaluation** — analysing detection results and training performance.
5. **Computer vision pipelines** — combining object detection with OCR.
6. **Inference optimisation** — processing images programmatically and measuring execution time.
7. **Software engineering** — separating training, inference and utility functionality into different modules.

## Future Improvements

Potential improvements include:

* Increasing the size and diversity of the training dataset
* Improving class balance between boat types
* Adding more sailing classes
* Systematic evaluation using precision, recall and mAP
* Hyperparameter optimisation
* Comparing YOLOv8 and YOLO11 performance
* Exporting the trained model to ONNX or TensorFlow Lite
* Optimising inference for Raspberry Pi / edge hardware
* Improving boat-number OCR accuracy
* Real-time detection from a camera feed

## Licence

This project is licensed under the **GNU General Public License v3.0**.

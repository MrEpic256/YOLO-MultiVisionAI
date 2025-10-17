# 🎯 YOLO-MultiVisionAI

YOLOv8 Object Detection Training Pipeline on Microsoft COCO Dataset.

## 📊 About the Project

A comprehensive project for training YOLOv8 models on 80 object classes from the COCO 2017 dataset.

- **Model**: YOLOv8n (nano)
- **Dataset**: Microsoft COCO 2017 (116,408 train / 5,000 val)
- **Classes**: 80 (people, vehicles, animals, objects)
- **Framework**: Ultralytics YOLOv8

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MrEpic256/YOLO-MultiVisionAI.git
cd YOLO-MultiVisionAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the [Microsoft COCO Dataset](https://universe.roboflow.com/jacob-solawetz/microsoft-coco/dataset/2) in YOLOv8 format and extract it to the `Microsoft COCO.v2-raw.yolov8/` folder.

### 4. Verify Dataset

```bash
python check_dataset.py
```

### 5. Start Training

```bash
python train_yolov8.py
```

or (for Windows):

```bash
.\start_training.bat
```

### 6. Test the Model

```bash
python test_yolov8.py
```

## 📁 Project Structure

```
YOLO-MultiVisionAI/
├── train_yolov8.py          # Training script
├── test_yolov8.py           # Testing script
├── check_dataset.py         # Dataset verification
├── coco_dataset.yaml        # Dataset configuration
├── requirements.txt         # Python dependencies
├── start_training.bat       # Windows batch file
└── README.md                # Documentation
```

## 🎯 Training Parameters

- **Epochs**: 10 (can be changed in `train_yolov8.py`)
- **Batch size**: 16
- **Image size**: 640x640
- **Data augmentation**: yes (HSV, flip, translate, scale, mosaic)
- **Device**: GPU (CUDA) if available, otherwise CPU

## 📊 Training Results

After training, models are saved in:

```
runs/detect/coco_yolov8_train/
├── weights/
│   ├── best.pt      # Best model
│   └── last.pt      # Last epoch
└── results.png      # Metrics plots
```

### Expected Metrics (10 epochs)

- **mAP50**: ~43%
- **mAP50-95**: ~30%
- **Precision**: ~57%
- **Recall**: ~40%

## 💻 Using the Trained Model

```python
from ultralytics import YOLO

# Load model
model = YOLO('runs/detect/coco_yolov8_train/weights/best.pt')

# Detect on image
results = model('image.jpg')
results[0].show()

# Detect on video
results = model('video.mp4')
```

## 📦 System Requirements

**Minimum:**
- Python 3.8+
- 8 GB RAM
- 10 GB free disk space

**Recommended:**
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM
- CUDA 11.8+

## 📚 Detailed Documentation

See [README_YOLO_TRAINING.md](README_YOLO_TRAINING.md) for detailed information about:
- Training parameters
- Data augmentation
- Troubleshooting
- Model optimization

## 📝 License

- **Code**: MIT License
- **COCO Dataset**: CC BY 4.0

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Microsoft COCO Dataset](https://cocodataset.org/)
- [Roboflow](https://roboflow.com/)

---

**Made with ❤️ for Object Detection**

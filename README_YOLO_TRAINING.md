# 🎯 YOLOv8 Object Detection Training Pipeline

A comprehensive guide for training YOLOv8 models on the Microsoft COCO dataset for object detection.

## 📊 Dataset

**Microsoft COCO 2017** - one of the largest object detection datasets:
- **80 classes**: people, vehicles, animals, household items, etc.
- **Train**: 116,408 images
- **Validation**: 5,000 images
- **Format**: YOLOv8 (normalized coordinates)

### Object Classes (examples):
- **Transportation**: car, bicycle, motorbike, bus, truck, train, boat, aeroplane
- **People**: person
- **Animals**: dog, cat, horse, cow, sheep, elephant, bear, zebra, giraffe
- **Objects**: chair, sofa, bed, bottle, cup, laptop, cell phone, book
- **Food**: banana, apple, pizza, sandwich, orange, hot dog, cake

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Dataset

```bash
python check_dataset.py
```

This script will verify the dataset structure and display statistics.

### 3. Train the Model

```bash
python train_yolov8.py
```

**Training Parameters:**
- Model: `yolov8n.pt` (nano - fast, lightweight)
- Epochs: 10
- Batch size: 16
- Image size: 640x640
- Device: GPU (if available) or CPU
- Data augmentation: yes (rotation, scaling, brightness)

**Alternative Models** (can be changed in `train_yolov8.py`):
- `yolov8n.pt` - Nano (fastest, smallest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (recommended)
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate, slowest)

### 4. Test the Model

```bash
python test_yolov8.py
```

The script will perform:
- ✅ Validation on the full validation dataset
- ✅ Testing on 10 random images
- ✅ Visualization of results
- ✅ Generation of metrics report

## 📁 Project Structure

```
YOLO-MultiVisionAI/
│
├── Microsoft COCO.v2-raw.yolov8/    # Dataset
│   ├── train/
│   │   ├── images/                   # Training images
│   │   └── labels/                   # YOLO format annotations
│   ├── valid/
│   │   ├── images/                   # Validation images
│   │   └── labels/                   # YOLO format annotations
│   └── data.yaml                     # Original configuration
│
├── coco_dataset.yaml                 # Training configuration
├── train_yolov8.py                   # Training script
├── test_yolov8.py                    # Testing script
├── check_dataset.py                  # Dataset verification
├── requirements.txt                  # Dependencies
│
└── runs/detect/coco_yolov8_train/   # Training results
    ├── weights/
    │   ├── best.pt                   # 🏆 Best model
    │   └── last.pt                   # Last epoch
    ├── results.png                   # Metrics plots
    ├── confusion_matrix.png          # Confusion matrix
    └── ...                           # Other plots
```

## 📊 Quality Metrics

After training, you will obtain the following metrics:

- **mAP50** - Mean Average Precision @ IoU=0.5
- **mAP50-95** - Mean Average Precision @ IoU=0.5:0.95 (main COCO metric)
- **Precision** - How many predicted objects are actually correct
- **Recall** - How many objects from the dataset were found

### Expected Results for YOLOv8n:
- mAP50-95: ~37-40%
- mAP50: ~53-56%
- Speed: ~100+ FPS on RTX 3080

## 🎨 Results Visualization

After testing (`test_yolov8.py`), the `test_results/` folder will contain:
- 10 images with detected objects (bounding boxes)
- `summary_grid.png` - grid with 4 examples
- Console report with top-10 most frequent classes

## 🔧 Configuration

### Changing Training Parameters

Edit `train_yolov8.py`:

```python
config = {
    'model': 'yolov8m.pt',     # Change model
    'epochs': 150,             # More epochs
    'batch': 32,               # Larger batch (requires more GPU memory)
    'imgsz': 640,              # Image size
    'patience': 50,            # Early stopping
}
```

### Data Augmentation

The following augmentations are configured in `train_yolov8.py`:
- Horizontal flip (50%)
- HSV color (hue, saturation, value)
- Translation (10%)
- Scale (50%)
- Mosaic (100%)

## 💾 System Requirements

**Minimum:**
- Python 3.8+
- 8 GB RAM
- 10 GB free disk space

**Recommended:**
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM
- CUDA 11.8+
- 20 GB free disk space

## 🐛 Troubleshooting

### Error: Out of Memory (OOM)
Reduce the `batch` size in `train_yolov8.py`:
```python
'batch': 8,  # or even 4
```

### Error: CUDA not available
Training will run on CPU (slow). To speed up:
1. Install CUDA Toolkit
2. Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Slow Training
- Use a smaller model (`yolov8n.pt`)
- Reduce `imgsz` to 416 or 320
- Reduce the number of `workers`

## 📚 Additional Resources

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

## 📝 License

- **Code**: MIT License
- **COCO Dataset**: CC BY 4.0

---

**Happy Training! 🚀**

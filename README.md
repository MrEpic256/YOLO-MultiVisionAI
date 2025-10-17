# 🎯 YOLO-MultiVisionAI

YOLOv8 Object Detection Training Pipeline на датасеті Microsoft COCO.

## 📊 Про проект

Проект для навчання моделі YOLOv8 на 80 класах об'єктів з датасету COCO 2017.

- **Модель**: YOLOv8n (nano)
- **Датасет**: Microsoft COCO 2017 (116,408 train / 5,000 val)
- **Класів**: 80 (люди, автомобілі, тварини, предмети)
- **Framework**: Ultralytics YOLOv8

## 🚀 Швидкий старт

### 1. Клонувати репозиторій

```bash
git clone https://github.com/YOUR_USERNAME/YOLO-MultiVisionAI.git
cd YOLO-MultiVisionAI
```

### 2. Встановити залежності

```bash
pip install -r requirements.txt
```

### 3. Завантажити датасет

Завантажте [Microsoft COCO Dataset](https://universe.roboflow.com/jacob-solawetz/microsoft-coco/dataset/2) у форматі YOLOv8 та розпакуйте в папку `Microsoft COCO.v2-raw.yolov8/`

### 4. Перевірити датасет

```bash
python check_dataset.py
```

### 5. Запустити навчання

```bash
python train_yolov8.py
```

або (для Windows):

```bash
.\start_training.bat
```

### 6. Тестування моделі

```bash
python test_yolov8.py
```

## 📁 Структура проекту

```
YOLO-MultiVisionAI/
├── train_yolov8.py          # Скрипт навчання
├── test_yolov8.py           # Скрипт тестування
├── check_dataset.py         # Перевірка датасету
├── coco_dataset.yaml        # Конфігурація датасету
├── requirements.txt         # Залежності Python
├── start_training.bat       # Батч-файл для Windows
└── README.md                # Документація
```

## 🎯 Параметри навчання

- **Епохи**: 10 (можна змінити в `train_yolov8.py`)
- **Batch size**: 16
- **Розмір зображення**: 640x640
- **Data augmentation**: так (HSV, flip, translate, scale, mosaic)
- **Device**: GPU (CUDA) якщо доступний, інакше CPU

## 📊 Результати навчання

Після навчання моделі зберігаються в:

```
runs/detect/coco_yolov8_train/
├── weights/
│   ├── best.pt      # Найкраща модель
│   └── last.pt      # Остання епоха
└── results.png      # Графіки метрик
```

### Очікувані метрики (10 епох):

- **mAP50**: ~43%
- **mAP50-95**: ~30%
- **Precision**: ~57%
- **Recall**: ~40%

## 💻 Використання навченої моделі

```python
from ultralytics import YOLO

# Завантажити модель
model = YOLO('runs/detect/coco_yolov8_train/weights/best.pt')

# Детекція на зображенні
results = model('image.jpg')
results[0].show()

# Детекція на відео
results = model('video.mp4')
```

## 📦 Системні вимоги

**Мінімальні:**
- Python 3.8+
- 8 GB RAM
- 10 GB вільного місця

**Рекомендовані:**
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU з 6+ GB VRAM
- CUDA 11.8+

## 📚 Детальна документація

Дивіться [README_YOLO_TRAINING.md](README_YOLO_TRAINING.md) для детальної інформації про:
- Параметри навчання
- Data augmentation
- Troubleshooting
- Оптимізацію моделі

## 📝 Ліцензія

- **Код**: MIT License
- **COCO Dataset**: CC BY 4.0

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Microsoft COCO Dataset](https://cocodataset.org/)
- [Roboflow](https://roboflow.com/)

---

**Made with ❤️ for Object Detection**

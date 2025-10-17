# 🎯 YOLOv8 Object Detection Training Pipeline

Проект для навчання моделі YOLOv8 на датасеті Microsoft COCO для детекції об'єктів.

## 📊 Датасет

**Microsoft COCO 2017** - один з найбільших датасетів для детекції об'єктів:
- **80 класів**: люди, автомобілі, тварини, предмети побуту тощо
- **Train**: 116,408 зображень
- **Validation**: 5,000 зображень
- **Формат**: YOLOv8 (нормалізовані координати)

### Класи об'єктів (приклади):
- **Транспорт**: car, bicycle, motorbike, bus, truck, train, boat, aeroplane
- **Люди**: person
- **Тварини**: dog, cat, horse, cow, sheep, elephant, bear, zebra, giraffe
- **Предмети**: chair, sofa, bed, bottle, cup, laptop, cell phone, book
- **Їжа**: banana, apple, pizza, sandwich, orange, hot dog, cake

## 🚀 Швидкий старт

### 1. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 2. Перевірка датасету

```bash
python check_dataset.py
```

Цей скрипт перевірить структуру датасету та виведе статистику.

### 3. Навчання моделі

```bash
python train_yolov8.py
```

**Параметри навчання:**
- Модель: `yolov8n.pt` (nano - швидка, легка)
- Епохи: 100
- Batch size: 16
- Розмір зображення: 640x640
- Device: GPU (якщо доступний) або CPU
- Data augmentation: так (обертання, масштабування, яскравість)

**Альтернативні моделі** (можна змінити в `train_yolov8.py`):
- `yolov8n.pt` - Nano (найшвидша, найменша)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (рекомендується)
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (найточніша, найповільніша)

### 4. Тестування моделі

```bash
python test_yolov8.py
```

Скрипт виконає:
- ✅ Валідацію на повному valid датасеті
- ✅ Тестування на 10 випадкових зображеннях
- ✅ Візуалізацію результатів
- ✅ Створення звіту з метриками

## 📁 Структура проекту

```
YOLO-MultiVisionAI/
│
├── Microsoft COCO.v2-raw.yolov8/    # Датасет
│   ├── train/
│   │   ├── images/                   # Тренувальні зображення
│   │   └── labels/                   # Анотації YOLO format
│   ├── valid/
│   │   ├── images/                   # Валідаційні зображення
│   │   └── labels/                   # Анотації YOLO format
│   └── data.yaml                     # Оригінальна конфігурація
│
├── coco_dataset.yaml                 # Конфігурація для навчання
├── train_yolov8.py                   # Скрипт навчання
├── test_yolov8.py                    # Скрипт тестування
├── check_dataset.py                  # Перевірка датасету
├── requirements.txt                  # Залежності
│
└── runs/detect/coco_yolov8_train/   # Результати навчання
    ├── weights/
    │   ├── best.pt                   # 🏆 Найкраща модель
    │   └── last.pt                   # Остання епоха
    ├── results.png                   # Графіки метрик
    ├── confusion_matrix.png          # Матриця помилок
    └── ...                           # Інші графіки
```

## 📊 Метрики якості

Після навчання ви отримаєте такі метрики:

- **mAP50** - Mean Average Precision @ IoU=0.5
- **mAP50-95** - Mean Average Precision @ IoU=0.5:0.95 (основна метрика COCO)
- **Precision** - Точність (скільки з передбачених об'єктів справді правильні)
- **Recall** - Повнота (скільки об'єктів з датасету знайдено)

### Очікувані результати для YOLOv8n:
- mAP50-95: ~37-40%
- mAP50: ~53-56%
- Швидкість: ~100+ FPS на RTX 3080

## 🎨 Візуалізація результатів

Після тестування (`test_yolov8.py`) у папці `test_results/` з'являться:
- 10 зображень з виявленими об'єктами (bounding boxes)
- `summary_grid.png` - grid з 4 прикладами
- Звіт у консолі з топ-10 найчастіших класів

## 🔧 Налаштування

### Зміна параметрів навчання

Відредагуйте `train_yolov8.py`:

```python
config = {
    'model': 'yolov8m.pt',     # Змінити модель
    'epochs': 150,             # Більше епох
    'batch': 32,               # Більший batch (потребує більше GPU пам'яті)
    'imgsz': 640,              # Розмір зображення
    'patience': 50,            # Ранній стоп
}
```

### Data Augmentation

У `train_yolov8.py` налаштовані такі augmentation:
- Horizontal flip (50%)
- HSV колір (hue, saturation, value)
- Translation (10%)
- Scale (50%)
- Mosaic (100%)

## 💾 Системні вимоги

**Мінімальні:**
- Python 3.8+
- 8 GB RAM
- 10 GB вільного місця на диску

**Рекомендовані:**
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU з 6+ GB VRAM
- CUDA 11.8+
- 20 GB вільного місця на диску

## 🐛 Troubleshooting

### Помилка: Out of Memory (OOM)
Зменшіть `batch` size у `train_yolov8.py`:
```python
'batch': 8,  # або навіть 4
```

### Помилка: CUDA not available
Навчання відбудеться на CPU (повільно). Для прискорення:
1. Встановіть CUDA Toolkit
2. Встановіть PyTorch з CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Повільне навчання
- Використайте меншу модель (`yolov8n.pt`)
- Зменшіть `imgsz` до 416 або 320
- Зменшіть кількість `workers`

## 📚 Додаткові ресурси

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

## 📝 Ліцензія

- **Код**: MIT License
- **COCO Dataset**: CC BY 4.0

---

**Успіхів у навчанні! 🚀**

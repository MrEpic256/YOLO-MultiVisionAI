"""
🎯 Скрипт для навчання моделі YOLOv8 на датасеті Microsoft COCO
Детекція об'єктів: 80 класів (люди, автомобілі, тварини, предмети та ін.)
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml

def check_environment():
    """Перевірка середовища та доступності GPU"""
    print("=" * 60)
    print("🔍 ПЕРЕВІРКА СЕРЕДОВИЩА")
    print("=" * 60)
    
    # Перевірка CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступний: {'✅ Так' if cuda_available else '❌ Ні'}")
    
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA версія: {torch.version.cuda}")
        print(f"Пам'ять GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ Навчання буде виконуватися на CPU (повільно)")
    
    print()
    return cuda_available

def train_model():
    """Основна функція навчання"""
    
    # Перевірка середовища
    use_gpu = check_environment()
    
    # Параметри навчання
    config = {
        'model': 'yolov8n.pt',  # nano модель (можна змінити на yolov8m.pt або yolov8l.pt)
        'data': 'coco_dataset.yaml',
        'epochs': 10,
        'imgsz': 640,
        'batch': 16,
        'device': 0 if use_gpu else 'cpu',
        'workers': 8,
        'project': 'runs/detect',
        'name': 'coco_yolov8_train',
        'patience': 50,  # ранній стоп після 50 епох без покращення
        'save': True,
        'save_period': 10,  # зберігати чекпоінт кожні 10 епох
        'verbose': True,
        'plots': True,
    }
    
    print("=" * 60)
    print("📋 ПАРАМЕТРИ НАВЧАННЯ")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:15s}: {value}")
    print()
    
    # Перевірка конфігураційного файлу
    if not Path(config['data']).exists():
        print(f"❌ Помилка: файл {config['data']} не знайдено!")
        return
    
    # Завантаження конфігурації датасету
    with open(config['data'], 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    print("=" * 60)
    print("📊 ІНФОРМАЦІЯ ПРО ДАТАСЕТ")
    print("=" * 60)
    print(f"Датасет: Microsoft COCO 2017")
    print(f"Кількість класів: {dataset_config['nc']}")
    print(f"Тренувальний набір: {config['data']}")
    print(f"Валідаційний набір: {config['data']}")
    print()
    
    # Ініціалізація моделі
    print("=" * 60)
    print("🚀 ПОЧАТОК НАВЧАННЯ")
    print("=" * 60)
    print(f"Завантаження моделі: {config['model']}")
    
    try:
        model = YOLO(config['model'])
        
        # Налаштування data augmentation
        augmentation = {
            'hsv_h': 0.015,      # Hue augmentation
            'hsv_s': 0.7,        # Saturation augmentation
            'hsv_v': 0.4,        # Value augmentation
            'degrees': 0.0,      # Rotation (+/- deg)
            'translate': 0.1,    # Translation (+/- fraction)
            'scale': 0.5,        # Scale (+/- gain)
            'shear': 0.0,        # Shear (+/- deg)
            'perspective': 0.0,  # Perspective (+/- fraction)
            'flipud': 0.0,       # Vertical flip (probability)
            'fliplr': 0.5,       # Horizontal flip (probability)
            'mosaic': 1.0,       # Mosaic augmentation (probability)
            'mixup': 0.0,        # MixUp augmentation (probability)
        }
        
        # Навчання моделі
        results = model.train(
            data=config['data'],
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=config['device'],
            workers=config['workers'],
            project=config['project'],
            name=config['name'],
            patience=config['patience'],
            save=config['save'],
            save_period=config['save_period'],
            verbose=config['verbose'],
            plots=config['plots'],
            **augmentation
        )
        
        print()
        print("=" * 60)
        print("✅ НАВЧАННЯ ЗАВЕРШЕНО!")
        print("=" * 60)
        
        # Отримуємо реальний шлях де YOLOv8 зберіг моделі (може бути з цифрою: train, train2, train3...)
        save_dir = Path(results.save_dir)
        best_model = save_dir / 'weights' / 'best.pt'
        last_model = save_dir / 'weights' / 'last.pt'
        
        print(f"📁 Результати збережено в: {save_dir}")
        print(f"🏆 Найкраща модель: {best_model}")
        print(f"📊 Останній чекпоінт: {last_model}")
        print()
        
        # Валідація моделі
        print("=" * 60)
        print("📊 ВАЛІДАЦІЯ МОДЕЛІ")
        print("=" * 60)
        
        metrics = model.val()
        
        print(f"\n📈 Метрики якості:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        print()
        
        return save_dir, best_model
        
    except Exception as e:
        print(f"❌ Помилка під час навчання: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🎯 YOLOv8 Object Detection Training Pipeline        ║
    ║          Microsoft COCO Dataset (80 classes)             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    save_dir, best_model = train_model()
    
    if best_model and best_model.exists():
        print("=" * 60)
        print("🎉 УСПІХ!")
        print("=" * 60)
        print(f"Ваша модель готова до використання: {best_model}")
        print("\nДля тестування використайте скрипт test_yolov8.py")
    else:
        print("❌ Навчання не вдалося. Перевірте помилки вище.")

"""
Скрипт для перевірки структури датасету COCO YOLOv8
"""
import os
from pathlib import Path

dataset_path = r"c:\Users\tankc\YOLO-MultiVisionAI\Microsoft COCO.v2-raw.yolov8"

print("=== Перевірка структури датасету ===\n")

# Перевірка основних папок
for split in ['train', 'valid']:
    split_path = Path(dataset_path) / split
    
    print(f"\n📁 {split.upper()}:")
    
    # Перевірка labels
    labels_path = split_path / 'labels'
    if labels_path.exists():
        label_files = list(labels_path.glob('*'))
        txt_files = [f for f in label_files if f.suffix == '.txt']
        img_files = [f for f in label_files if f.suffix in ['.jpg', '.jpeg', '.png']]
        
        print(f"  Labels папка: {len(label_files)} файлів")
        print(f"    - .txt файли: {len(txt_files)}")
        print(f"    - зображення: {len(img_files)}")
        
        if label_files:
            print(f"  Приклад файлу: {label_files[0].name}")
    
    # Перевірка images
    images_path = split_path / 'images'
    if images_path.exists():
        image_files = list(images_path.glob('*'))
        print(f"  Images папка: {len(image_files)} файлів")
        
        if image_files:
            print(f"  Приклад файлу: {image_files[0].name}")

# Перевірка, чи зображення всередині labels
print("\n\n=== Аналіз файлів у labels ===")
labels_path = Path(dataset_path) / 'train' / 'labels'
all_files = list(labels_path.glob('*'))[:20]  # Перші 20 файлів

extensions = {}
for f in all_files:
    ext = f.suffix
    extensions[ext] = extensions.get(ext, 0) + 1
    
print(f"Розширення файлів (перші 20):")
for ext, count in extensions.items():
    print(f"  {ext}: {count}")

print("\nПриклади файлів:")
for f in all_files[:5]:
    print(f"  {f.name}")

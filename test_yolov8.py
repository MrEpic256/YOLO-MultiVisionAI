"""
🧪 Скрипт для тестування та візуалізації результатів YOLOv8
Демонстрація детекції об'єктів на зображеннях
"""

import os
import random
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def find_test_images(dataset_path, num_images=10):
    """Знаходить випадкові зображення для тестування"""
    valid_images_path = Path(dataset_path) / 'valid' / 'images'
    
    if not valid_images_path.exists():
        print(f"❌ Папка {valid_images_path} не знайдена!")
        return []
    
    # Отримуємо всі зображення
    image_files = list(valid_images_path.glob('*.jpg'))
    
    if len(image_files) == 0:
        print(f"❌ Зображення не знайдені в {valid_images_path}")
        return []
    
    # Вибираємо випадкові зображення
    num_images = min(num_images, len(image_files))
    selected_images = random.sample(image_files, num_images)
    
    print(f"✅ Знайдено {len(image_files)} зображень, вибрано {num_images} для тестування")
    
    return selected_images

def visualize_predictions(model, image_paths, save_dir='test_results'):
    """Візуалізація предикцій на зображеннях"""
    
    # Створення папки для результатів
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 60)
    print("🎨 ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ")
    print("=" * 60)
    
    results_list = []
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] Обробка: {img_path.name}")
        
        # Виконання детекції
        results = model(str(img_path), verbose=False)[0]
        
        # Збереження результату
        output_path = save_path / f"result_{idx}_{img_path.name}"
        
        # Малюємо bbox на зображенні
        img_with_boxes = results.plot()
        
        # Зберігаємо
        cv2.imwrite(str(output_path), img_with_boxes)
        
        # Інформація про детекції
        num_detections = len(results.boxes)
        print(f"  ✅ Знайдено об'єктів: {num_detections}")
        
        if num_detections > 0:
            # Виводимо інформацію про кожен об'єкт
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results.names[cls]
                print(f"    - {class_name}: {conf:.2%}")
        
        results_list.append({
            'path': img_path,
            'output_path': output_path,
            'num_objects': num_detections,
            'results': results
        })
    
    print(f"\n💾 Результати збережено в: {save_path}")
    
    return results_list

def create_summary_report(results_list, model_path):
    """Створення звіту про результати"""
    
    print("\n" + "=" * 60)
    print("📊 ЗВІТ ПРО ТЕСТУВАННЯ")
    print("=" * 60)
    
    print(f"\n🤖 Модель: {model_path}")
    print(f"📸 Кількість тестових зображень: {len(results_list)}")
    
    # Статистика по об'єктах
    total_objects = sum(r['num_objects'] for r in results_list)
    avg_objects = total_objects / len(results_list) if results_list else 0
    
    print(f"\n📈 Статистика детекції:")
    print(f"  Всього знайдено об'єктів: {total_objects}")
    print(f"  Середньо на зображення: {avg_objects:.2f}")
    
    # Розподіл по класах
    class_counts = {}
    for r in results_list:
        results = r['results']
        for box in results.boxes:
            cls = int(box.cls[0])
            class_name = results.names[cls]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        print(f"\n🏷️ Топ-10 найчастіших класів:")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, count) in enumerate(sorted_classes[:10], 1):
            print(f"  {i:2d}. {class_name:20s}: {count:3d} виявлень")

def display_sample_results(results_list, num_samples=4):
    """Відображення прикладів результатів у grid"""
    
    num_samples = min(num_samples, len(results_list))
    
    if num_samples == 0:
        print("⚠️ Немає результатів для відображення")
        return
    
    print(f"\n🖼️ Відображення {num_samples} прикладів...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx in range(num_samples):
        img_path = results_list[idx]['output_path']
        img = Image.open(img_path)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f"Зображення {idx+1}: {results_list[idx]['num_objects']} об'єктів", 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    # Приховуємо порожні підграфіки
    for idx in range(num_samples, 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results/summary_grid.png', dpi=150, bbox_inches='tight')
    print("✅ Grid збережено: test_results/summary_grid.png")
    
    try:
        plt.show()
    except:
        print("⚠️ Не вдалося відобразити вікно (можливо, запуск без GUI)")

def main():
    """Основна функція тестування"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          🧪 YOLOv8 Model Testing & Visualization        ║
    ║              Object Detection on COCO Dataset            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Шлях до моделі
    model_path = 'runs/detect/coco_yolov8_train/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Модель не знайдена: {model_path}")
        print("⚠️ Спочатку навчіть модель, запустивши: python train_yolov8.py")
        return
    
    print(f"✅ Завантаження моделі: {model_path}")
    model = YOLO(model_path)
    
    # Валідація моделі на всьому valid set
    print("\n" + "=" * 60)
    print("📊 ВАЛІДАЦІЯ НА ПОВНОМУ ДАТАСЕТІ")
    print("=" * 60)
    
    try:
        metrics = model.val(data='coco_dataset.yaml')
        
        print("\n📈 Метрики моделі:")
        print(f"  {'Метрика':<20s} {'Значення':>10s}")
        print("  " + "-" * 32)
        print(f"  {'mAP50':<20s} {metrics.box.map50:>10.4f}")
        print(f"  {'mAP50-95':<20s} {metrics.box.map:>10.4f}")
        print(f"  {'Precision':<20s} {metrics.box.mp:>10.4f}")
        print(f"  {'Recall':<20s} {metrics.box.mr:>10.4f}")
        print(f"  {'F1-Score':<20s} {(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-6)):>10.4f}")
        
    except Exception as e:
        print(f"⚠️ Помилка під час валідації: {e}")
    
    # Тестування на випадкових зображеннях
    dataset_path = r"c:\Users\tankc\YOLO-MultiVisionAI\Microsoft COCO.v2-raw.yolov8"
    test_images = find_test_images(dataset_path, num_images=10)
    
    if not test_images:
        print("❌ Не знайдено зображень для тестування")
        return
    
    # Візуалізація результатів
    results_list = visualize_predictions(model, test_images)
    
    # Звіт
    create_summary_report(results_list, model_path)
    
    # Відображення прикладів
    display_sample_results(results_list, num_samples=4)
    
    print("\n" + "=" * 60)
    print("✅ ТЕСТУВАННЯ ЗАВЕРШЕНО!")
    print("=" * 60)
    print("📁 Перегляньте папку 'test_results' для результатів")

if __name__ == "__main__":
    main()

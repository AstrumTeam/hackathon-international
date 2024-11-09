import cv2
from ultralytics import YOLO
import os

# Путь к изображению и к обученной модели
image_path = "yolo_dataset_processed/val/images/82.jpg"  # Замените на путь к вашему изображению
model_path = "/Users/vladislav/Хакатоны/hackathon-international/best.pt"  # Замените на путь к обученной модели YOLO
output_folder = "cropped_images"  # Папка для сохранения обрезанных изображений

# Создание папки, если она не существует
os.makedirs(output_folder, exist_ok=True)

# Загрузка модели YOLO
model = YOLO(model_path)

# Загрузка изображения
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Изображение по пути {image_path} не найдено.")

# Получение детекций объектов
results = model.predict(source=image, save=False, save_txt=False, conf=0.7)  # conf=0.5 - минимальная вероятность обнаружения

height, width, _ = image.shape

# Преобразование результатов в удобный формат и обрезка изображений
for i, result in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Координаты рамки (xmin, ymin, xmax, ymax)
    class_id = int(result.cls[0])  # ID класса
    label = model.names[class_id]  # Получение метки класса

    # Вычисление центра и размеров рамки
    x_center = (x1 + x2) / (2 * width)
    y_center = (y1 + y2) / (2 * height)
    bbox_width = (x2 - x1) / width
    bbox_height = (y2 - y1) / height

    # Форматированный вывод координат
    print(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Обрезка изображения по найденной рамке
    cropped_image = image[y1:y2, x1:x2]  # Обрезаем изображение

    # Сохранение обрезанного изображения
    output_path = os.path.join(output_folder, f"cropped_{i + 1}.jpg")  # Используем i + 1 для нумерации
    cv2.imwrite(output_path, cropped_image)
    print(f"Обрезанное изображение сохранено по пути: {output_path}")

# result = results[0].boxes[0]
# x1, y1, x2, y2 = map(int, result.xyxy[0])  # Координаты рамки (xmin, ymin, xmax, ymax)
# class_id = int(result.cls[0])  # ID класса
# label = model.names[class_id]  # Получение метки класса

# # Вычисление центра и размеров рамки
# x_center = (x1 + x2) / (2 * width)
# y_center = (y1 + y2) / (2 * height)
# bbox_width = (x2 - x1) / width
# bbox_height = (y2 - y1) / height

# # Форматированный вывод координат
# print(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

# # Обрезка изображения по найденной рамке
# cropped_image = image[y1:y2, x1:x2]  # Обрезаем изображение

# # Сохранение обрезанного изображения
# output_path = os.path.join(output_folder, f"cropped.jpg")  # Используем i + 1 для нумерации
# cv2.imwrite(output_path, cropped_image)
# print(f"Обрезанное изображение сохранено по пути: {output_path}")
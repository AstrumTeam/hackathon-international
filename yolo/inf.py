import cv2
from ultralytics import YOLO

# Путь к изображению и к обученной модели
image_path = "67.jpg"  # Замените на путь к вашему изображению
model_path = "first_iteration_processed.pt"    # Замените на путь к обученной модели YOLO

# Загрузка модели YOLO
model = YOLO(model_path)

# Загрузка изображения
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Изображение по пути {image_path} не найдено.")

# Получение детекций объектов
results = model.predict(source=image, save=False, save_txt=False, conf=0.5)  # conf=0.5 - минимальная вероятность обнаружения

# Преобразование результатов в удобный формат и отрисовка рамок
for result in results[0].boxes:
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Координаты рамки (xmin, ymin, xmax, ymax)
    conf = result.conf[0]                      # Уверенность предсказания
    label = result.cls[0]                      # Класс объекта
    
    # Отрисовка рамки и подписи
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зеленая рамка
    text = f"{label}: {conf:.2f}"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Отображение изображения с детекцией
cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Если нужно сохранить изображение с детекциями
output_path = "output_image.jpg"
cv2.imwrite(output_path, image)
print(f"Результат сохранен по пути: {output_path}")
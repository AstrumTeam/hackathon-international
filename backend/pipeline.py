import easyocr
import cv2
from ultralytics import YOLO
import os

def process_image(image_path, output_folder, model_path='model_yolo.pt', model_ocr_path='model_easyocr.pth'):
    # Загружаем модели
    model_ocr = easyocr.Reader(['ru', 'en'], gpu=False, recognizer=model_ocr_path)
    model = YOLO(model_path)
    
    # Загружаем изображение
    image = cv2.imread(image_path)

    # Преобразуем изображение в оттенки серого и обрабатываем его
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_image = cv2.medianBlur(gray_img, 5)
    image_preprocess = cv2.Laplacian(median_image, cv2.THRESH_BINARY, ksize=5, scale=0.55)
    image_preprocess = cv2.cvtColor(image_preprocess, cv2.COLOR_GRAY2BGR)

    # Применяем модель YOLO к изображению
    results = model.predict(source=image_preprocess, save=False, save_txt=False, conf=0.8)
    
    recognized_text = []
    height, width, _ = image.shape
    
    # Перебираем все найденные рамки объектов
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        class_id = int(result.cls[0])
        label = model.names[class_id]
        
        # Рисуем рамку и метку на изображении
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Обрезаем изображение по рамке
        cropped_image = image[y1:y2, x1:x2]
        
        # Распознаём текст в обрезанном изображении с помощью easyocr
        result = model_ocr.readtext(
            cropped_image, 
            rotation_info=[90, 180, 270], 
            detail=0, 
            batch_size=32, 
            text_threshold=0.5, 
            low_text=0.2,
            allowlist='139-08ТС.456Пр27АМГ/чотНЕОВLPЗИКРвБAMлмаксионOZI[]бКУЭmaxхinФ'
        )
        
        recognized_text.extend(result)
    
    # Сохраняем изображение с нарисованными границами
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, f"processed_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image)
    
    # Возвращаем путь к сохранённому изображению и распознанный текст
    return {"recognized_text": " ".join(recognized_text), "output_image_path": output_image_path}
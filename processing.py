import os
import cv2

# Путь к папке с изображениями и аннотациями
input_folder = "train Росатом/train/imgs"
output_folder = "image_preprocessed"  # Укажите нужный путь к папке для сохранения результатов

# Создаем выходную папку, если её нет
os.makedirs(output_folder, exist_ok=True)

# Функция для обработки и сохранения выделенных областей
def process_image(image_path, output_folder):
    # Открываем изображение
    image = cv2.imread(image_path)
    
    # Преобразуем изображение в оттенки серого
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применяем медианный фильтр
    median_image = cv2.medianBlur(gray_img, 5)
    
    # Применяем фильтр Лапласа для выделения границ
    laplac = cv2.Laplacian(median_image, cv2.THRESH_BINARY, scale=0.55, ksize=5)
    
    # Генерируем уникальное имя файла для каждой области
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = f"{base_name}.jpg"
    
    # Сохраняем изображение в указанной выходной папке
    cv2.imwrite(os.path.join(output_folder, output_file), laplac)
    print(f"Сохранено: {os.path.join(output_folder, output_file)}")

# Проходим по всем изображениям в папке ввода
for image_name in os.listdir(input_folder):
    # Определяем путь к изображению
    image_path = os.path.join(input_folder, image_name)
    
    # Проверяем, что это файл изображения (по расширению, например, .jpg, .png)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        process_image(image_path, output_folder)

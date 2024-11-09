import os
import random
import shutil
from pathlib import Path

# Путь к исходным папкам с изображениями и метками
images_path = "image_preprocessed"
labels_path = "/Users/vladislav/Хакатоны/hackathon-international/train Росатом/train/labels"

# Папка, куда будет сохранен новый YOLO-совместимый датасет
output_dir = "yolo_dataset_processed"
os.makedirs(output_dir, exist_ok=True)

# Процент данных для train, val и test
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Убедимся, что суммы пропорций равны 1.0
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Суммы пропорций должны составлять 1.0"

# Список всех изображений и меток с приведением расширения и имени к нижнему регистру
image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
label_files = sorted([f for f in os.listdir(labels_path) if f.lower().endswith('.txt')])

# Найдем общие базовые имена файлов, чтобы получить пары (приводим к нижнему регистру)
image_files_set = {Path(f).stem.lower() for f in image_files}
label_files_set = {Path(f).stem.lower() for f in label_files}
common_files = image_files_set & label_files_set

# Отфильтруем файлы, оставив только пары с совпадающими именами (регистр теперь не имеет значения)
image_files = [f for f in image_files if Path(f).stem.lower() in common_files]
label_files = [f for f in label_files if Path(f).stem.lower() in common_files]

# Подсчитаем, сколько файлов не имеют пары
missing_images = len(label_files_set - image_files_set)
missing_labels = len(image_files_set - label_files_set)
print(f"Файлов меток без соответствующих изображений: {missing_images}")
print(f"Файлов изображений без соответствующих меток: {missing_labels}")

# Убедимся, что теперь количество изображений и меток совпадает
assert len(image_files) == len(label_files), "Количество изображений и меток должно совпадать"

# Перемешиваем данные и делим их на train, val и test
data = list(zip(image_files, label_files))
random.shuffle(data)

train_split = int(len(data) * train_ratio)
val_split = train_split + int(len(data) * val_ratio)

train_data = data[:train_split]
val_data = data[train_split:val_split]
test_data = data[val_split:]

# Функция для копирования файлов
def copy_files(data, split_name):
    image_dir = os.path.join(output_dir, split_name, "images")
    label_dir = os.path.join(output_dir, split_name, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for image_file, label_file in data:
        shutil.copy(os.path.join(images_path, image_file), os.path.join(image_dir, image_file))
        shutil.copy(os.path.join(labels_path, label_file), os.path.join(label_dir, label_file))

# Копируем файлы в соответствующие папки
copy_files(train_data, "train")
copy_files(val_data, "val")
copy_files(test_data, "test")

# Создаем YAML файл для конфигурации YOLO
yaml_content = f"""
train: {output_dir}/train/images
val: {output_dir}/val/images
test: {output_dir}/test/images

# Количество классов
nc: <количество_классов>

# Имена классов (замените на свои классы)
names: [<имя_класса_1>, <имя_класса_2>, ...]
"""

with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

print("Датасет успешно подготовлен!")
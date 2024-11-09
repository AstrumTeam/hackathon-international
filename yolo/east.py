import cv2
import numpy as np

# Параметры модели
east_model_path = 'yolo_test/frozen_east_text_detection.pb'  # Укажите путь к файлу модели
min_confidence = 0.5  # Минимальная уверенность для отбора текстовых областей
input_width = 640  # Размер входного изображения для модели
input_height = 640

# Функция для обработки и обнаружения текста в кадре
def detect_text_in_frame(frame):
    orig = frame.copy()
    (H, W) = frame.shape[:2]

    # Подготовка изображения для подачи в модель EAST
    blob = cv2.dnn.blobFromImage(frame, 1.0, (input_width, input_height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

    # Постобработка для получения ограничивающих рамок
    rectangles, confidences = decode_predictions(scores, geometry, min_confidence)
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    # Отображение рамок на кадре
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * W / input_width)
        startY = int(startY * H / input_height)
        endX = int(endX * W / input_width)
        endY = int(endY * H / input_height)
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return orig

# Декодирование предсказаний (функция аналогична предыдущему примеру)
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            # Смещение для координат
            offsetX, offsetY = x * 4.0, y * 4.0

            # Угол и синус-косинус
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Высчитываем ширину и высоту рамки
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Рассчитываем начальные и конечные координаты рамки
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rectangles.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rectangles, confidences

# Non-max suppression (NMS) для удаления избыточных рамок
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Инициализация модели EAST
net = cv2.dnn.readNet(east_model_path)

# Запуск видеопотока с веб-камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Обнаружение текста
    frame_with_text = detect_text_in_frame(frame)

    # Отображение кадра с рамками
    cv2.imshow("Text Detection", frame_with_text)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
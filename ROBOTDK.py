import cv2
from ultralytics import YOLO

# Параметры калибровки
KNOWN_DISTANCE = 50.0  # см, расстояние до яблока при калибровке
KNOWN_WIDTH = 8.0      # см, реальная ширина яблока

# Функция для вычисления фокусного расстояния камеры
def find_focal_length(known_distance, known_width, width_in_pixels):
    return (width_in_pixels * known_distance) / known_width

# Функция для вычисления расстояния до объекта
def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Загрузите модель YOLOv8 (замените на свой, если есть)
model = YOLO("yolov8n.pt")

# Сначала нужно вручную измерить ширину яблока в пикселях на фото с известным расстоянием
# Например, на calibration.jpg яблоко на 50 см, ширина в пикселях = 100 (замените на своё значение)
width_in_pixels_calibration = 100  # примерное значение, заменить на реальное

# Вычисляем фокусное расстояние
focal_length_found = find_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, width_in_pixels_calibration)
print(f"Фокусное расстояние камеры: {focal_length_found:.2f}")

# Открываем камеру
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

print("Нажмите 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра")
        break

    # Детекция объектов
    results = model.predict(frame, conf=0.5)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box.astype(int)
            class_id = int(cls)
            class_name = model.model.names[class_id].lower()

            # Интересуемся только яблоками
            if class_name == "apple":
                # Ширина объекта в пикселях
                obj_width = x2 - x1

                # Вычисляем расстояние
                distance = distance_to_camera(KNOWN_WIDTH, focal_length_found, obj_width)

                # Рисуем прямоугольник и выводим расстояние
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {distance:.2f} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Детекция и расстояние до яблока", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

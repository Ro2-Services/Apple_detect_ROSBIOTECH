import cv2
import requests
import matplotlib.pyplot as plt

# Recover the model
ROBOFLOW_API_KEY = "vV7dnPsE0Xx1xol0oZjx"
MODEL_ID = "apple-fruit-ajaxe-a17eg"
VERSION = "1"

# Model URL
url = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={ROBOFLOW_API_KEY}"

# Загрузить изображение
image_path = "pomme2.jpg"
image = cv2.imread(image_path)

# Проверьте правильность загрузки изображения
if image is None:
    raise ValueError("Erreur : Unable to load image. Check file path.")

# Преобразование в формат, используемый API
_, img_encoded = cv2.imencode(".jpg", image)
files = {"file": img_encoded.tobytes()}

# Отправить в Roboflow и получить результаты
response = requests.post(url, files=files)
detections = response.json()
print(detections)

# Проектирование коробок обнаружения
for detection in detections["predictions"]:
    x, y, width, height = detection["x"], detection["y"], detection["width"], detection["height"]

    # Проверьте размеры изображения
    print(image.shape)

    # Дизайн коробок и этикетки
    cv2.rectangle(image,
                  (int(x - width // 2), int(y - height // 2)),
                  (int(x + width // 2), int(y + height // 2)),
                  (0, 255, 0), 2)

    cv2.putText(image, detection["class"], (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#  Решение 1 - Используйте Matplotlib для отображения изображения
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование BGR → RGB
plt.figure(figsize=(8,6))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Apple detection")
plt.show()

# Решение 2 - правильная запись изображения
cv2.imwrite("detection_result.jpg", image)
print("The image is saved under the name 'detection_result.jpg', results.")
